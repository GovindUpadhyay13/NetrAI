"""
bus/consumers/reid_matcher.py
Cross-camera Re-ID consumer.
On incident detection, registers the subject crop in the Qdrant gallery.
On fresh footage from secondary cameras, checks detected subjects against the gallery;
on match, publishes a 'reid_match' event tagged with the originating incident_id.

Assumptions:
- Uses SubjectEmbedder (YOLOv8 + shared CLIP encoder) and QdrantReIDGallery.
- Cosine similarity match threshold is typically 0.65 - 0.75.
"""

from typing import List, Optional
import numpy as np
from PIL import Image

from bus.publisher import EventBus
from bus.schemas import StageEnum, SurveillanceEvent
from reid.embed import SubjectEmbedder
from reid.gallery import QdrantReIDGallery


class ReIDMatcherConsumer:
    def __init__(
        self,
        bus: Optional[EventBus] = None,
        embedder: Optional[SubjectEmbedder] = None,
        gallery: Optional[QdrantReIDGallery] = None,
        similarity_threshold: float = 0.68,
    ):
        self.bus = bus or EventBus()
        self.embedder = embedder or SubjectEmbedder()
        self.gallery = gallery or QdrantReIDGallery()
        self.similarity_threshold = similarity_threshold

        self.matches_found = []

        # Listen to event bus
        self.bus.subscribe_memory(self.handle_event)

    def handle_event(self, event: SurveillanceEvent):
        """Processes events: if flagged/analyzed with crops, indexes into gallery."""
        pass  # indexing is triggered directly when incident frames are flagged

    def index_incident_subject(
        self,
        incident_id: str,
        camera_id: str,
        frames_bgr: List[np.ndarray],
    ) -> int:
        """
        Extracts subject crops from incident frames and enrolls them into Qdrant gallery.
        """
        indexed_count = 0
        for i, frame in enumerate(frames_bgr[::max(1, len(frames_bgr) // 5)]):
            crops_info = self.embedder.detect_and_crop(frame)
            if not crops_info:
                # If YOLO doesn't detect a bounding box, use full center crop as fallback
                H, W = frame.shape[:2]
                center_bgr = frame[int(H * 0.1) : int(H * 0.9), int(W * 0.2) : int(W * 0.8)]
                import cv2
                crops_info = [(Image.fromarray(cv2.cvtColor(center_bgr, cv2.COLOR_BGR2RGB)), "person", [0, 0, W, H])]

            crop_images = [c[0] for c in crops_info]
            embeddings = self.embedder.embed_crops(crop_images)

            for crop_img, (c_img, c_name, bbox), emb in zip(crop_images, crops_info, embeddings):
                self.gallery.add_flagged_subject(
                    incident_id=incident_id,
                    camera_id=camera_id,
                    embedding=emb.tolist(),
                    subject_type=c_name,
                    payload_ref=f"bbox_{bbox}",
                )
                indexed_count += 1

        print(f"[ReIDMatcher] Enrolled {indexed_count} subject crops for Incident {incident_id[:8]} into gallery.")
        return indexed_count

    def scan_camera_feed_for_matches(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        timestamp: str = "",
    ) -> List[dict]:
        """
        Scans a frame from another camera, detects subjects, queries Qdrant,
        and publishes 'reid_match' events upon detection.
        """
        crops_info = self.embedder.detect_and_crop(frame_bgr)
        if not crops_info:
            # Fallback center crop if detection is faint
            H, W = frame_bgr.shape[:2]
            center_bgr = frame_bgr[int(H * 0.1) : int(H * 0.9), int(W * 0.2) : int(W * 0.8)]
            import cv2
            crops_info = [(Image.fromarray(cv2.cvtColor(center_bgr, cv2.COLOR_BGR2RGB)), "person", [0, 0, W, H])]

        crop_images = [c[0] for c in crops_info]
        embeddings = self.embedder.embed_crops(crop_images)

        matches = []
        for (crop_img, c_name, bbox), emb in zip(crops_info, embeddings):
            match_res = self.gallery.match_subject(emb.tolist(), score_threshold=self.similarity_threshold)
            if match_res:
                orig_inc_id = match_res["origin_incident_id"]
                orig_cam = match_res["origin_camera_id"]
                score = match_res["score"]

                print(f"\n[ReIDMatcher] >>> CROSS-CAMERA MATCH DETECTED! <<<")
                print(f"  Originating Incident: {orig_inc_id[:8]} (Cam: {orig_cam})")
                print(f"  New Sighting Camera:  {camera_id} | Similarity: {score:.3f}")

                # Publish Stage: 'reid_match'
                reid_event = SurveillanceEvent(
                    incident_id=orig_inc_id,
                    camera_id=camera_id,
                    stage=StageEnum.REID_MATCH,
                    anomaly_score=score,
                    anomaly_type="Cross-Camera Subject Re-ID",
                    distress_gesture=False,
                    vlm_report=f"Subject sighted on {camera_id} (Re-ID similarity: {score:.2f}, origin: {orig_cam})",
                    severity=None,
                    payload_ref=f"bbox_{bbox}",
                )
                self.bus.publish(reid_event)
                matches.append(match_res)
                self.matches_found.append(match_res)

        return matches
