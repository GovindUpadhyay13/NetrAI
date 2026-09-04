"""
bus/consumers/trace_logger.py
Event bus consumer that records every incoming surveillance event into the SQLite trace database.

Assumptions:
- Listens to all stages ('anomaly_detected', 'gesture_flagged', 'vlm_analyzed', 'dispatched', 'reid_match', 'trace').
- Writes rows asynchronously or synchronously into SQLite.
"""

import threading
import time
from typing import Optional
from bus.publisher import EventBus
from bus.schemas import SurveillanceEvent
from trace.db import DEFAULT_DB_PATH, init_db, insert_event


class TraceLoggerConsumer:
    def __init__(self, bus: Optional[EventBus] = None, db_path: str = DEFAULT_DB_PATH):
        self.bus = bus or EventBus()
        self.db_path = db_path
        init_db(self.db_path)
        self.is_running = False
        self._thread = None

        # Register direct memory subscriber
        self.bus.subscribe_memory(self.handle_event)

    def handle_event(self, event: SurveillanceEvent):
        """Processes and logs a single event into SQLite."""
        try:
            row_id = insert_event(event, self.db_path)
            print(f"[TraceLogger] Logged event #{row_id} | Stage: {event.stage} | Incident: {event.incident_id[:8]}... | Cam: {event.camera_id}")
        except Exception as e:
            print(f"[TraceLogger] Failed to write event to DB: {e}")

    def start_redis_worker(self, group_name: str = "trace_loggers", consumer_name: str = "worker-1"):
        """Background worker thread consuming from Redis Streams if connected."""
        if not self.bus.is_connected_redis:
            return

        self.is_running = True

        def _worker():
            r = self.bus.redis_client
            stream = self.bus.stream_key
            try:
                r.xgroup_create(stream, group_name, id="0", mkstream=True)
            except Exception:
                pass  # Group already exists

            while self.is_running:
                try:
                    entries = r.xreadgroup(group_name, consumer_name, {stream: ">"}, count=10, block=1000)
                    if not entries:
                        continue
                    for _, messages in entries:
                        for msg_id, data in messages:
                            event = SurveillanceEvent.from_redis_dict(data)
                            self.handle_event(event)
                            r.xack(stream, group_name, msg_id)
                except Exception as e:
                    time.sleep(0.5)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        print(f"[TraceLogger] Started Redis Streams worker thread.")

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
