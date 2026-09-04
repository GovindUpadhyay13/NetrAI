"""
reid/gallery.py
Qdrant vector store gallery for storing and matching flagged subject Re-ID embeddings.

Assumptions:
- Uses Qdrant vector database (in-memory or local persistent storage).
- Vector dimension is 512 (CLIP ViT-B/16).
- Distance metric is COSINE.
"""

import os
from typing import Dict, List, Optional
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantReIDGallery:
    def __init__(
        self,
        collection_name: str = "surveillance_gallery",
        qdrant_url: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.vector_dim = 512

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        elif storage_path:
            os.makedirs(storage_path, exist_ok=True)
            self.client = QdrantClient(path=storage_path)
        else:
            # High-speed in-memory gallery
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            print(f"[QdrantGallery] Created collection: {self.collection_name} (512-dim COSINE)")

    def add_flagged_subject(
        self,
        incident_id: str,
        camera_id: str,
        embedding: List[float],
        subject_type: str = "person",
        payload_ref: str = "",
        extra_meta: Optional[Dict] = None,
    ) -> str:
        """Stores a flagged subject embedding linked to an originating incident_id."""
        point_id = str(uuid4())
        payload = {
            "incident_id": incident_id,
            "origin_camera_id": camera_id,
            "subject_type": subject_type,
            "payload_ref": payload_ref,
            **(extra_meta or {}),
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        print(f"[QdrantGallery] Stored flagged subject {point_id[:8]} from Incident {incident_id[:8]} (Cam: {camera_id})")
        return point_id

    def match_subject(
        self,
        query_embedding: List[float],
        score_threshold: float = 0.70,
        exclude_incident_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Searches the gallery for nearest matching subject.
        Returns match dict if cosine similarity >= score_threshold, else None.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=1,
            score_threshold=score_threshold,
        )

        if results and results.points:
            top = results.points[0]
            # Verify not matching itself from the same camera/instant if required
            match_data = {
                "match_id": str(top.id),
                "score": round(float(top.score), 4),
                "origin_incident_id": top.payload.get("incident_id"),
                "origin_camera_id": top.payload.get("origin_camera_id"),
                "subject_type": top.payload.get("subject_type"),
                "payload_ref": top.payload.get("payload_ref"),
            }
            return match_data

        return None
