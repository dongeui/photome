"""Local-first durable processing pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.contracts import DerivedAssetKind, MediaFaceInput, MediaKind, MediaTagInput, ProcessingJobKind, ProcessingJobState
from app.models.job import ProcessingJob
from app.models.media import MediaFile
from app.models.person import Person
from app.models.tag import Tag
from app.services.analysis import FaceAnalysisError, FaceAnalysisService
from app.services.analysis import auto_tags, image_signals
from app.services.embedding import clip as clip_embedding
from app.services.fingerprint.service import FingerprintService
from app.services.metadata.service import MetadataService
from app.services.ocr import extract as extract_ocr
from app.services.processing.incremental import IncrementalScanService, IncrementalScanSummary
from app.services.processing.registry import MediaCatalog
from app.services.scanner.service import ScannerService
from app.services.semantic import SemanticCatalog
from app.services.thumbnail.service import ThumbnailService
from app.services.video.service import VideoKeyframeService


logger = logging.getLogger(__name__)

PLACE_TAG_TYPES = frozenset({"place", "location", "place_detail"})
PERSON_TAG_TYPES = frozenset({"person", "people", "face"})


@dataclass(frozen=True)
class PipelineSummary:
    job_id: str
    job_kind: str
    status: str
    payload: dict[str, Any] | None
    result: dict[str, Any] | None
    error_stage: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class FaceMaterializationResult:
    faces: tuple[MediaFaceInput, ...]
    person_tags: tuple[MediaTagInput, ...]
    summaries: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()


@dataclass
class PersonCentroidState:
    person: Person
    centroid: tuple[float, ...]
    sample_count: int
    relative_path: Path


class ProcessingPipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        scanner: ScannerService,
        fingerprint_service: FingerprintService,
        metadata_service: MetadataService,
        thumbnail_service: ThumbnailService,
        keyframe_service: VideoKeyframeService,
        *,
        face_analysis_service: FaceAnalysisService | None = None,
        derived_root: Path | None = None,
        embeddings_root: Path | None = None,
        face_match_threshold: float = 0.363,
        place_tag_precision: int = 3,
        semantic_ocr_enabled: bool = True,
        semantic_clip_enabled: bool = False,
        semantic_ocr_version: str = "ocr-v1",
        semantic_embedding_version: str = "embedding-v1",
        semantic_auto_tag_version: str = "auto-v1",
    ) -> None:
        self._session_factory = session_factory
        self._scanner = scanner
        self._fingerprint_service = fingerprint_service
        self._metadata_service = metadata_service
        self._thumbnail_service = thumbnail_service
        self._keyframe_service = keyframe_service
        self._face_analysis_service = face_analysis_service
        self._derived_root = (derived_root or Path("./derived_root")).expanduser().resolve()
        self._embeddings_root = (embeddings_root or (self._derived_root / "embeddings")).expanduser().resolve()
        self._face_match_threshold = max(-1.0, min(1.0, face_match_threshold))
        self._place_tag_precision = max(0, place_tag_precision)
        self._semantic_ocr_enabled = semantic_ocr_enabled
        self._semantic_clip_enabled = semantic_clip_enabled
        self._semantic_ocr_version = semantic_ocr_version
        self._semantic_embedding_version = semantic_embedding_version
        self._semantic_auto_tag_version = semantic_auto_tag_version

    def submit_scan_job(self, *, full_scan: bool = False, run_now: bool = True, trigger: str = "manual") -> PipelineSummary:
        with self._session_factory() as session:
            job = ProcessingJob(
                job_kind=ProcessingJobKind.SCAN.value,
                status=ProcessingJobState.QUEUED.value,
                payload_json={"full_scan": full_scan, "trigger": trigger},
                attempts=0,
            )
            session.add(job)
            session.flush()

            if run_now:
                try:
                    self._run_scan_job(session, job, full_scan=full_scan)
                except Exception:
                    session.commit()
                    return self._to_summary(job)

            session.commit()
            return self._to_summary(job)

    def run_scan_job(self, job_id: str) -> PipelineSummary:
        with self._session_factory() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise ValueError(f"Unknown job_id: {job_id}")
            if job.job_kind != ProcessingJobKind.SCAN.value:
                raise ValueError(f"Job {job_id} is not a scan job")
            full_scan = bool((job.payload_json or {}).get("full_scan"))
            try:
                self._run_scan_job(session, job, full_scan=full_scan)
            except Exception:
                session.commit()
                return self._to_summary(job)
            session.commit()
            return self._to_summary(job)

    def rebuild_media_assets(self, file_id: str) -> PipelineSummary:
        with self._session_factory() as session:
            job = ProcessingJob(
                job_kind=ProcessingJobKind.PIPELINE.value,
                status=ProcessingJobState.RUNNING.value,
                payload_json={"file_id": file_id, "asset_refresh": True},
                attempts=1,
                started_at=datetime.utcnow(),
            )
            session.add(job)
            session.flush()

            catalog = MediaCatalog(session)
            media_file = catalog.get_media(file_id)
            if media_file is None:
                job.status = ProcessingJobState.FAILED.value
                job.error_stage = "catalog"
                job.error_message = "media file not found"
                job.finished_at = datetime.utcnow()
                session.commit()
                return self._to_summary(job)

            try:
                result = self._refresh_media_assets(session, media_file)
                job.status = ProcessingJobState.SUCCEEDED.value
                job.result_json = result
                job.finished_at = datetime.utcnow()
            except Exception as exc:
                logger.exception("failed to refresh media assets", extra={"file_id": file_id})
                job.status = ProcessingJobState.FAILED.value
                job.error_stage = "asset_pipeline"
                job.error_message = str(exc)
                job.finished_at = datetime.utcnow()

            session.commit()
            return self._to_summary(job)

    def run_semantic_backfill(self, *, batch_size: int = 50) -> dict[str, Any]:
        """Generate CLIP embeddings for media that were processed before CLIP was enabled."""
        if not self._semantic_clip_enabled:
            return {"skipped": True, "reason": "clip_disabled", "pending": 0, "succeeded": 0, "failed": 0}

        with self._session_factory() as session:
            catalog = MediaCatalog(session)
            pending = catalog.list_media_needing_embedding(limit=batch_size)
            semantic_catalog = SemanticCatalog(session)
            succeeded = failed = 0

            for media_file in pending:
                try:
                    embedding_result = self._materialize_clip_embedding(media_file)
                    if embedding_result:
                        semantic_catalog.register_embedding(media_file.file_id, **embedding_result)
                        clip_rel = Path(embedding_result["embedding_ref"])
                        catalog.register_derived_asset(
                            media_file.file_id,
                            DerivedAssetKind.CLIP_EMBEDDING,
                            clip_rel,
                            version=embedding_result["version"],
                            content_type="application/octet-stream",
                        )
                        embedding_tags = auto_tags.tags_from_embedding_file(
                            embedding_result["embedding_ref"],
                            self._embeddings_root,
                        )
                        if embedding_tags:
                            existing_auto_tags = [
                                MediaTagInput(tag_type=t.tag_type, tag_value=t.tag_value)
                                for t in media_file.tags
                                if t.tag_type == "auto"
                            ]
                            merged = auto_tags.merge_auto_tags(existing_auto_tags, embedding_tags)
                            catalog.replace_tags_for_types(media_file.file_id, ["auto"], merged)
                            semantic_catalog.upsert_auto_tag_state(
                                media_file.file_id,
                                tags=merged,
                                version=self._semantic_auto_tag_version,
                            )
                        succeeded += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.warning(
                        "semantic backfill failed",
                        extra={"file_id": media_file.file_id, "error": str(exc)},
                    )
                    failed += 1

            session.commit()
            return {
                "skipped": False,
                "pending": len(pending),
                "succeeded": succeeded,
                "failed": failed,
                "has_more": len(pending) == batch_size,
            }

    def status_snapshot(self) -> dict[str, Any]:
        with self._session_factory() as session:
            catalog = MediaCatalog(session)
            job_counts = self._job_counts(session)
            return {
                "media": {
                    "total": catalog.count_media(),
                    "status_counts": catalog.media_status_counts(),
                    "kind_counts": catalog.media_kind_counts(),
                    "errors": catalog.count_media(status="error"),
                    "waiting_stable": catalog.count_observations(status="waiting_stable"),
                    "missing": catalog.count_media(status="missing"),
                    "observations": catalog.observation_status_counts(),
                },
                "jobs": job_counts,
            }

    def _run_scan_job(self, session: Session, job: ProcessingJob, *, full_scan: bool) -> IncrementalScanSummary:
        now = datetime.utcnow()
        job.status = ProcessingJobState.RUNNING.value
        job.started_at = job.started_at or now
        job.attempts = (job.attempts or 0) + 1
        job.error_stage = None
        job.error_message = None
        session.flush()

        try:
            scan_summary = IncrementalScanService(
                self._scanner,
                self._fingerprint_service,
                self._metadata_service,
            ).run(session)
            processed_summary = self._process_pending_media(session, trigger_job_id=job.id)
        except Exception as exc:
            logger.exception("scan job failed", extra={"job_id": job.id})
            job.status = ProcessingJobState.FAILED.value
            job.error_stage = "scan"
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            session.flush()
            raise

        job.status = ProcessingJobState.SUCCEEDED.value
        job.result_json = {
            "full_scan": full_scan,
            "summary": asdict(scan_summary),
            "processed": processed_summary,
        }
        job.finished_at = datetime.utcnow()
        return scan_summary

    def _process_pending_media(self, session: Session, *, trigger_job_id: str) -> dict[str, Any]:
        catalog = MediaCatalog(session)
        pending_media = catalog.list_media_for_processing()
        succeeded = 0
        failed = 0

        for media_file in pending_media:
            job = ProcessingJob(
                job_kind=ProcessingJobKind.PIPELINE.value,
                status=ProcessingJobState.RUNNING.value,
                payload_json={"file_id": media_file.file_id, "trigger_job_id": trigger_job_id},
                attempts=1,
                started_at=datetime.utcnow(),
            )
            session.add(job)
            session.flush()

            try:
                result = self._refresh_media_assets(session, media_file)
                job.status = ProcessingJobState.SUCCEEDED.value
                job.result_json = result
                job.finished_at = datetime.utcnow()
                succeeded += 1
            except Exception as exc:
                logger.exception("failed to refresh media assets", extra={"file_id": media_file.file_id})
                catalog.mark_media_error(media_file.file_id, stage="asset_pipeline", message=str(exc), now=datetime.utcnow())
                job.status = ProcessingJobState.FAILED.value
                job.error_stage = "asset_pipeline"
                job.error_message = str(exc)
                job.finished_at = datetime.utcnow()
                failed += 1

        session.flush()
        return {
            "pending": len(pending_media),
            "succeeded": succeeded,
            "failed": failed,
        }

    def _refresh_media_assets(self, session: Session, media_file: MediaFile) -> dict[str, Any]:
        catalog = MediaCatalog(session)
        result: dict[str, Any] = {"file_id": media_file.file_id, "assets": []}
        preserved_tags, existing_place_tags, existing_person_tags = self._split_existing_tags(media_file.tags)
        preserved_tags = [tag for tag in preserved_tags if tag.tag_type != "auto"]
        place_tags = self._materialize_place_tags(media_file)
        person_tags = existing_person_tags
        auto_tag_inputs: list[MediaTagInput] = []
        face_result: FaceMaterializationResult | None = None
        analysis_warnings: list[str] = []
        media_kind = media_file.media_kind

        if media_kind == MediaKind.IMAGE.value:
            location = self._thumbnail_service.generate(Path(media_file.current_path), media_file.file_id, MediaKind.IMAGE)
            catalog.register_derived_asset(media_file.file_id, location.kind, location.relative_path)
            result["assets"].append({"kind": location.kind.value, "path": str(location.relative_path)})

            face_result = self._materialize_faces(session, media_file)
            if face_result is not None:
                catalog.upsert_faces(media_file.file_id, face_result.faces)
                person_tags = list(face_result.person_tags)
                result["faces"] = list(face_result.summaries)
                analysis_warnings.extend(face_result.warnings)
            semantic_result = self._materialize_image_semantics(session, media_file)
            if semantic_result:
                auto_tag_inputs = semantic_result.pop("_auto_tag_inputs", [])
                result["semantic"] = semantic_result
            catalog.set_media_status(media_file.file_id, status="thumb_done", now=datetime.utcnow())

        elif media_kind == MediaKind.VIDEO.value:
            thumb_location = self._thumbnail_service.generate(Path(media_file.current_path), media_file.file_id, MediaKind.VIDEO)
            catalog.register_derived_asset(media_file.file_id, thumb_location.kind, thumb_location.relative_path)
            result["assets"].append({"kind": thumb_location.kind.value, "path": str(thumb_location.relative_path)})
            keyframe_locations = self._keyframe_service.extract(Path(media_file.current_path), media_file.file_id)
            for location in keyframe_locations:
                catalog.register_derived_asset(media_file.file_id, location.kind, location.relative_path)
                result["assets"].append({"kind": location.kind.value, "path": str(location.relative_path)})
            catalog.set_media_status(media_file.file_id, status="analysis_done", now=datetime.utcnow())

        else:
            raise ValueError(f"Unsupported media kind: {media_kind}")

        tags = catalog.upsert_tags(media_file.file_id, preserved_tags + place_tags + person_tags + auto_tag_inputs)
        if tags:
            result["tags"] = [{"type": tag.tag_type, "value": tag.tag_value} for tag in tags]
        elif place_tags or existing_place_tags or person_tags or existing_person_tags:
            result["tags"] = []

        if analysis_warnings:
            result["analysis_warnings"] = analysis_warnings
        return result

    def _materialize_image_semantics(self, session: Session, media_file: MediaFile) -> dict[str, Any]:
        semantic_catalog = SemanticCatalog(session)
        result: dict[str, Any] = {}
        source_path = self._analysis_source_path(media_file)
        ocr_text = ""

        if self._semantic_ocr_enabled:
            ocr_result = extract_ocr(str(source_path))
            semantic_catalog.upsert_ocr(media_file.file_id, ocr_result, version=self._semantic_ocr_version)
            ocr_text = ocr_result.text
            result["ocr"] = {
                "engine": ocr_result.engine,
                "text_length": len(ocr_result.text),
                "blocks": len(ocr_result.blocks),
            }

        signal_payload = image_signals.extract_analysis(str(source_path), ocr_text)
        semantic_catalog.upsert_analysis(media_file.file_id, signal_payload)
        result["analysis"] = signal_payload
        signal_tags = auto_tags.tags_from_signals(signal_payload, ocr_text)
        embedding_tags: list[MediaTagInput] = []

        if self._semantic_clip_enabled:
            embedding_result = self._materialize_clip_embedding(media_file)
            if embedding_result:
                semantic_catalog.register_embedding(media_file.file_id, **embedding_result)
                # Also track as a versioned DerivedAsset for catalog visibility
                clip_rel = Path(embedding_result["embedding_ref"])
                MediaCatalog(session).register_derived_asset(
                    media_file.file_id,
                    DerivedAssetKind.CLIP_EMBEDDING,
                    clip_rel,
                    version=embedding_result["version"],
                    content_type="application/octet-stream",
                )
                result["embedding"] = embedding_result
                embedding_tags = auto_tags.tags_from_embedding_file(
                    embedding_result["embedding_ref"],
                    self._embeddings_root,
                )

        generated_tags = auto_tags.merge_auto_tags(signal_tags, embedding_tags)
        semantic_catalog.upsert_auto_tag_state(
            media_file.file_id,
            tags=generated_tags,
            version=self._semantic_auto_tag_version,
        )
        if generated_tags:
            result["_auto_tag_inputs"] = generated_tags
            result["auto_tags"] = [
                {"type": tag.tag_type, "value": tag.tag_value}
                for tag in generated_tags
            ]

        return result

    def _analysis_source_path(self, media_file: MediaFile) -> Path:
        thumbnail_path = self._derived_root / self._clip_source_thumbnail_relative_path(media_file.file_id)
        if thumbnail_path.is_file():
            return thumbnail_path
        return Path(media_file.current_path)

    def _materialize_clip_embedding(self, media_file: MediaFile) -> dict[str, Any] | None:
        try:
            clip_embedding.ensure_models()
            payload = clip_embedding.encode_image(media_file.current_path)
        except Exception as exc:
            fallback_path = self._derived_root / self._clip_source_thumbnail_relative_path(media_file.file_id)
            try:
                payload = clip_embedding.encode_image(str(fallback_path))
            except Exception:
                logger.warning(
                    "clip embedding skipped",
                    extra={"file_id": media_file.file_id, "path": media_file.current_path, "reason": str(exc)},
                )
                return None

        vector = clip_embedding.embedding_from_bytes(payload)
        relative_path = self._clip_embedding_relative_path(media_file.file_id)
        absolute_path = self._embedding_absolute_path(relative_path)
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile("wb", delete=False, dir=absolute_path.parent) as handle:
                tmp_path = Path(handle.name)
                import numpy as np

                np.save(handle, vector.astype("float32"))
            tmp_path.replace(absolute_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return {
            "model_name": "ViT-B-32/openai",
            "version": self._semantic_embedding_version,
            "embedding_ref": str(relative_path),
            "dimensions": int(vector.size),
            "checksum": None,
        }

    def _materialize_place_tags(self, media_file: MediaFile) -> list[MediaTagInput]:
        metadata = media_file.metadata_json
        if not isinstance(metadata, dict):
            return []

        gps_payload = metadata.get("gps")
        if not isinstance(gps_payload, dict):
            return []

        latitude = _coerce_float(gps_payload.get("latitude"))
        longitude = _coerce_float(gps_payload.get("longitude"))
        if latitude is None or longitude is None:
            return []

        grouped = f"{latitude:.{self._place_tag_precision}f},{longitude:.{self._place_tag_precision}f}"
        detailed = f"{latitude:.7f},{longitude:.7f}"
        return [
            MediaTagInput(tag_type="place", tag_value=grouped),
            MediaTagInput(tag_type="place_detail", tag_value=detailed),
        ]

    def _materialize_faces(self, session: Session, media_file: MediaFile) -> FaceMaterializationResult | None:
        if self._face_analysis_service is None:
            return None

        try:
            analysis = self._face_analysis_service.analyze_image_file(Path(media_file.current_path))
        except FaceAnalysisError as exc:
            logger.warning(
                "face analysis skipped",
                extra={"file_id": media_file.file_id, "path": media_file.current_path, "reason": str(exc)},
            )
            return None

        centroid_states = self._load_person_centroids(session)
        face_inputs: list[MediaFaceInput] = []
        person_tags: dict[str, MediaTagInput] = {}
        summaries: list[dict[str, Any]] = []

        for face in analysis.faces:
            person, similarity, is_new_person = self._resolve_person(session, centroid_states, face.embedding)
            embedding_ref = self._write_face_embedding(
                media_file=media_file,
                face_index=face.face_index,
                person=person,
                bbox=_build_bbox_payload(face.bbox),
                embedding=face.embedding,
            )
            face_input = MediaFaceInput(
                bbox=_build_bbox_payload(face.bbox),
                embedding_ref=embedding_ref,
                person_id=person.id,
                person_display_name=person.display_name,
            )
            face_inputs.append(face_input)
            person_tags[person.display_name] = MediaTagInput(tag_type="person", tag_value=person.display_name)
            summaries.append(
                {
                    "face_index": face.face_index,
                    "person_id": person.id,
                    "person": person.display_name,
                    "embedding_ref": embedding_ref,
                    "bbox": face_input.bbox,
                    "match": "new" if is_new_person else "reused",
                    "similarity": round(similarity, 6) if similarity is not None else None,
                }
            )

        return FaceMaterializationResult(
            faces=tuple(face_inputs),
            person_tags=tuple(person_tags.values()),
            summaries=tuple(summaries),
            warnings=analysis.warnings,
        )

    def _load_person_centroids(self, session: Session) -> list[PersonCentroidState]:
        states: list[PersonCentroidState] = []
        people = session.scalars(select(Person).order_by(Person.id.asc())).all()
        for person in people:
            relative_path = self._person_centroid_relative_path(person.id)
            payload = self._read_json(relative_path)
            if not isinstance(payload, dict):
                continue
            centroid = _coerce_embedding(payload.get("embedding"))
            if centroid is None:
                continue
            sample_count = _coerce_int(payload.get("sample_count")) or 1
            states.append(
                PersonCentroidState(
                    person=person,
                    centroid=_normalize_embedding(centroid),
                    sample_count=max(1, sample_count),
                    relative_path=relative_path,
                )
            )
        return states

    def _resolve_person(
        self,
        session: Session,
        centroid_states: list[PersonCentroidState],
        embedding: tuple[float, ...],
    ) -> tuple[Person, float | None, bool]:
        normalized_embedding = _normalize_embedding(embedding)
        best_state: PersonCentroidState | None = None
        best_similarity = -1.0

        for state in centroid_states:
            similarity = _cosine_similarity(state.centroid, normalized_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_state = state

        if best_state is not None and best_similarity >= self._face_match_threshold:
            best_state.sample_count += 1
            best_state.centroid = _normalize_embedding(
                tuple(
                    (
                        (best_state.centroid[index] * (best_state.sample_count - 1))
                        + normalized_embedding[index]
                    )
                    / best_state.sample_count
                    for index in range(len(normalized_embedding))
                )
            )
            self._write_person_centroid(best_state)
            return best_state.person, best_similarity, False

        person = self._create_person(session)
        state = PersonCentroidState(
            person=person,
            centroid=normalized_embedding,
            sample_count=1,
            relative_path=self._person_centroid_relative_path(person.id),
        )
        centroid_states.append(state)
        self._write_person_centroid(state)
        return person, None, True

    def _create_person(self, session: Session) -> Person:
        person = Person(display_name=f"person-pending-{uuid4().hex}")
        session.add(person)
        session.flush()
        person.display_name = f"person-{person.id:06d}"
        session.flush()
        return person

    def _write_face_embedding(
        self,
        *,
        media_file: MediaFile,
        face_index: int,
        person: Person,
        bbox: dict[str, Any],
        embedding: tuple[float, ...],
    ) -> str:
        relative_path = self._face_embedding_relative_path(media_file.file_id, face_index)
        self._write_json(
            relative_path,
            {
                "file_id": media_file.file_id,
                "face_index": face_index,
                "person_id": person.id,
                "person": person.display_name,
                "bbox": bbox,
                "embedding": list(embedding),
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
        return str(relative_path)

    def _write_person_centroid(self, state: PersonCentroidState) -> None:
        self._write_json(
            state.relative_path,
            {
                "person_id": state.person.id,
                "person": state.person.display_name,
                "sample_count": state.sample_count,
                "embedding": list(state.centroid),
                "updated_at": datetime.utcnow().isoformat(),
            },
        )

    def _read_json(self, relative_path: Path) -> dict[str, Any] | None:
        absolute_path = self._embedding_absolute_path(relative_path)
        if not absolute_path.is_file():
            return None
        try:
            with absolute_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_json(self, relative_path: Path, payload: dict[str, Any]) -> None:
        absolute_path = self._embedding_absolute_path(relative_path)
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=absolute_path.parent) as handle:
                tmp_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(absolute_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _face_embedding_relative_path(self, file_id: str, face_index: int) -> Path:
        shard = file_id[:2] if len(file_id) >= 2 else "xx"
        return Path("embeddings") / "faces" / "v1" / shard / f"{file_id}-face-{face_index:03d}.json"

    def _person_centroid_relative_path(self, person_id: int) -> Path:
        return Path("embeddings") / "people" / "v1" / f"person-{person_id:06d}.json"

    def _clip_embedding_relative_path(self, file_id: str) -> Path:
        shard = file_id[:2] if len(file_id) >= 2 else "xx"
        return Path("embeddings") / "clip" / self._semantic_embedding_version / shard / f"{file_id}.npy"

    def _clip_source_thumbnail_relative_path(self, file_id: str) -> Path:
        shard = file_id[:2] if len(file_id) >= 2 else "xx"
        return Path("thumb") / "v1" / shard / f"{file_id}.jpg"

    def _embedding_absolute_path(self, relative_path: Path) -> Path:
        try:
            suffix = relative_path.relative_to("embeddings")
        except ValueError:
            suffix = relative_path
        return self._embeddings_root / suffix

    def _split_existing_tags(
        self,
        tags: list[Tag],
    ) -> tuple[list[MediaTagInput], list[MediaTagInput], list[MediaTagInput]]:
        preserved: list[MediaTagInput] = []
        place_tags: list[MediaTagInput] = []
        person_tags: list[MediaTagInput] = []
        for tag in tags:
            tag_input = MediaTagInput(tag_type=tag.tag_type, tag_value=tag.tag_value)
            if tag.tag_type in PLACE_TAG_TYPES:
                place_tags.append(tag_input)
            elif tag.tag_type in PERSON_TAG_TYPES:
                person_tags.append(tag_input)
            else:
                preserved.append(tag_input)
        return preserved, place_tags, person_tags

    def _job_counts(self, session: Session) -> dict[str, Any]:
        status_rows = session.execute(
            select(ProcessingJob.status, func.count()).group_by(ProcessingJob.status)
        ).all()
        kind_rows = session.execute(
            select(ProcessingJob.job_kind, func.count()).group_by(ProcessingJob.job_kind)
        ).all()
        status_counts: dict[str, int] = {}
        kind_counts: dict[str, int] = {}
        for status, count in status_rows:
            status_counts[status] = int(count)
        for job_kind, count in kind_rows:
            kind_counts[job_kind] = int(count)
        return {"status_counts": status_counts, "kind_counts": kind_counts}

    def _to_summary(self, job: ProcessingJob) -> PipelineSummary:
        return PipelineSummary(
            job_id=job.id,
            job_kind=job.job_kind,
            status=job.status,
            payload=job.payload_json,
            result=job.result_json,
            error_stage=job.error_stage,
            error_message=job.error_message,
        )


def _build_bbox_payload(bbox: Any) -> dict[str, Any]:
    return {
        "x": int(bbox.x),
        "y": int(bbox.y),
        "width": int(bbox.width),
        "height": int(bbox.height),
        "confidence": float(bbox.confidence),
        "landmarks": [[float(x), float(y)] for x, y in bbox.landmarks],
    }


def _coerce_embedding(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    values: list[float] = []
    for item in value:
        coerced = _coerce_float(item)
        if coerced is None:
            return None
        values.append(coerced)
    if not values:
        return None
    return tuple(values)


def _normalize_embedding(embedding: tuple[float, ...]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in embedding))
    if magnitude <= 0.0:
        return embedding
    return tuple(value / magnitude for value in embedding)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude <= 0.0 or right_magnitude <= 0.0:
        return -1.0
    return sum(left[index] * right[index] for index in range(len(left))) / (left_magnitude * right_magnitude)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
