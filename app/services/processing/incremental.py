"""Incremental scan reconciliation logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging

from sqlalchemy.orm import Session

from app.services.fingerprint.service import FingerprintService
from app.services.metadata.service import MetadataService
from app.services.processing.registry import MediaCatalog
from app.services.scanner.service import ScannerService


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncrementalScanSummary:
    scanned: int = 0
    created: int = 0
    updated: int = 0
    moved: int = 0
    missing: int = 0
    failed: int = 0


class IncrementalScanService:
    def __init__(
        self,
        scanner: ScannerService,
        fingerprint_service: FingerprintService,
        metadata_service: MetadataService,
    ) -> None:
        self._scanner = scanner
        self._fingerprint_service = fingerprint_service
        self._metadata_service = metadata_service

    def run(self, session: Session) -> IncrementalScanSummary:
        catalog = MediaCatalog(session)
        scanned = created = updated = moved = failed = 0
        seen_paths: set[str] = set()
        active_source_roots = {
            str(source_root)
            for source_root in self._scanner.config.source_roots
            if source_root.exists()
        }
        now = datetime.utcnow()

        for scan_record in self._scanner.iter_files():
            scanned += 1
            seen_paths.add(str(scan_record.path))
            observation = catalog.observe_scan(
                scan_record,
                now=now,
                stability_window_seconds=self._scanner.config.stability_window_seconds,
            )
            if not observation.ready:
                continue

            try:
                identity = self._fingerprint_service.fingerprint(scan_record)
            except Exception as exc:
                failed += 1
                catalog.mark_observation_error(
                    scan_record,
                    stage="fingerprint",
                    message=str(exc),
                    now=now,
                    stability_window_seconds=self._scanner.config.stability_window_seconds,
                )
                logger.exception("failed to fingerprint scanned file", extra={"path": str(scan_record.path)})
                continue

            metadata_result = None
            metadata_error: str | None = None
            try:
                metadata_result = self._metadata_service.extract(scan_record)
            except Exception as exc:
                metadata_error = str(exc)

            change = catalog.upsert_media_file(
                scan_record,
                identity,
                metadata_result.metadata if metadata_result is not None else None,
                now=now,
            )
            if change.action == "created":
                created += 1
            elif change.action == "moved":
                moved += 1
                updated += 1
            elif change.action == "replaced":
                updated += 1
            elif change.action == "updated":
                updated += 1

            if metadata_error is not None:
                failed += 1
                catalog.mark_media_error(identity.file_id, stage="metadata", message=metadata_error, now=now)
                logger.error("failed to extract metadata for scanned file", extra={"path": str(scan_record.path)})
            elif metadata_result is not None and metadata_result.warnings:
                logger.debug(
                    "metadata warnings",
                    extra={"path": str(scan_record.path), "warnings": metadata_result.warnings},
                )

        if self._scanner.config.source_roots and not active_source_roots:
            logger.warning("all source roots unavailable; skipping missing reconciliation")
            missing = 0
        else:
            missing = catalog.mark_missing_except(seen_paths, active_source_roots)
        session.commit()
        return IncrementalScanSummary(
            scanned=scanned,
            created=created,
            updated=updated,
            moved=moved,
            missing=missing,
            failed=failed,
        )
