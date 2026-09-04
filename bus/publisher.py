"""
bus/publisher.py
Event bus publisher for Redis Streams with fallback in-memory pub-sub.

Assumptions:
- Stream key defaults to 'surveillance:events'.
- If Redis is reachable at REDIS_URL or localhost:6379, publishes with XADD.
- If Redis is offline, routes events through a synchronized thread-safe queue.
"""

import os
import queue
import threading
from typing import Callable, List, Optional
import redis

from .schemas import SurveillanceEvent


class EventBus:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        redis_url: Optional[str] = None,
        stream_key: str = "surveillance:events",
    ):
        if self._initialized:
            return

        self.stream_key = stream_key
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self.is_connected_redis = False

        # In-memory fallback
        self._memory_subscribers: List[Callable[[SurveillanceEvent], None]] = []
        self._memory_queue = queue.Queue()

        try:
            r = redis.from_url(self.redis_url, socket_timeout=1.0)
            r.ping()
            self.redis_client = r
            self.is_connected_redis = True
            print(f"[EventBus] Connected to live Redis at {self.redis_url} (Stream: {self.stream_key})")
        except Exception as e:
            print(f"[EventBus] Redis not available ({e}). Operating in resilient In-Memory Event Bus mode.")

        self._initialized = True

    def subscribe_memory(self, callback: Callable[[SurveillanceEvent], None]):
        """Subscribes an event handler (used in in-memory mode or for direct observer patterns)."""
        self._memory_subscribers.append(callback)

    def publish(self, event: SurveillanceEvent) -> str:
        """
        Publishes a surveillance event to Redis Streams (or in-memory bus).
        Returns the event/message ID.
        """
        event_dict = event.to_redis_dict()

        if self.is_connected_redis and self.redis_client:
            try:
                msg_id = self.redis_client.xadd(self.stream_key, event_dict)
                event_id = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
            except Exception as e:
                print(f"[EventBus] Redis XADD failed: {e}. Falling back to memory dispatch.")
                event_id = f"mem-{event.timestamp}"
        else:
            event_id = f"mem-{event.timestamp}"

        # Dispatch to registered consumers
        for subscriber in self._memory_subscribers:
            try:
                subscriber(event)
            except Exception as ex:
                print(f"[EventBus] Error in subscriber callback {subscriber}: {ex}")

        return event_id
