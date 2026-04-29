"""Local scheduler primitives for scan polling and daily full scans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from time import sleep
from typing import Optional

from app.core.settings import AppSettings
from app.services.processing.pipeline import ProcessingPipeline


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchedulerSnapshot:
    enabled: bool
    running: bool
    poll_interval_seconds: int
    daily_full_scan_hour: int
    daily_full_scan_minute: int
    semantic_scheduler_enabled: bool
    semantic_scheduler_interval_seconds: int
    last_poll_at: Optional[datetime]
    last_full_scan_at: Optional[datetime]
    next_poll_at: Optional[datetime]
    next_full_scan_at: Optional[datetime]
    last_semantic_maintenance_at: Optional[datetime]
    next_semantic_maintenance_at: Optional[datetime]


class SchedulerService:
    def __init__(self, settings: AppSettings, pipeline: ProcessingPipeline) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._enabled = settings.scheduler_enabled
        self._thread_enabled = settings.scheduler_enabled or settings.semantic_scheduler_enabled
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_poll_at: datetime | None = None
        self._last_full_scan_at: datetime | None = None
        self._last_semantic_maintenance_at: datetime | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._thread_enabled or self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="photome-scheduler", daemon=True)
        self._thread.start()
        self._running = True
        logger.info("scheduler started")

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._running = False
        logger.info("scheduler stopped")

    def tick(self, now: datetime | None = None) -> SchedulerSnapshot:
        now = now or datetime.utcnow()
        if self._enabled and self._is_poll_due(now):
            self._pipeline.submit_scan_job(full_scan=False, trigger="scheduler-poll")
            self._last_poll_at = now
        if self._enabled and self._is_full_scan_due(now):
            self._pipeline.submit_scan_job(full_scan=True, trigger="scheduler-full")
            self._last_full_scan_at = now
        if self._settings.semantic_scheduler_enabled and self._is_semantic_maintenance_due(now):
            self._pipeline.run_semantic_maintenance()
            self._last_semantic_maintenance_at = now
        return self.snapshot(now)

    def snapshot(self, now: datetime | None = None) -> SchedulerSnapshot:
        now = now or datetime.utcnow()
        return SchedulerSnapshot(
            enabled=self._enabled,
            running=self._running,
            poll_interval_seconds=self._settings.scheduler_poll_interval_seconds,
            daily_full_scan_hour=self._settings.scheduler_daily_full_scan_hour,
            daily_full_scan_minute=self._settings.scheduler_daily_full_scan_minute,
            semantic_scheduler_enabled=self._settings.semantic_scheduler_enabled,
            semantic_scheduler_interval_seconds=self._settings.semantic_scheduler_interval_seconds,
            last_poll_at=self._last_poll_at,
            last_full_scan_at=self._last_full_scan_at,
            next_poll_at=self._next_poll_at(now),
            next_full_scan_at=self._next_full_scan_at(now),
            last_semantic_maintenance_at=self._last_semantic_maintenance_at,
            next_semantic_maintenance_at=self._next_semantic_maintenance_at(now),
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("scheduler tick failed")
            self._stop_event.wait(max(1, self._settings.scheduler_poll_interval_seconds))

    def _is_poll_due(self, now: datetime) -> bool:
        next_poll = self._next_poll_at(now)
        return next_poll is None or now >= next_poll

    def _is_full_scan_due(self, now: datetime) -> bool:
        next_full_scan = self._next_full_scan_at(now)
        return next_full_scan is None or now >= next_full_scan

    def _next_poll_at(self, now: datetime) -> datetime | None:
        if self._last_poll_at is None:
            return now
        return self._last_poll_at + timedelta(seconds=max(1, self._settings.scheduler_poll_interval_seconds))

    def _next_full_scan_at(self, now: datetime) -> datetime | None:
        base = now.replace(
            hour=self._settings.scheduler_daily_full_scan_hour,
            minute=self._settings.scheduler_daily_full_scan_minute,
            second=0,
            microsecond=0,
        )
        if self._last_full_scan_at is None:
            return base if base >= now else base + timedelta(days=1)
        candidate = self._last_full_scan_at.replace(
            hour=self._settings.scheduler_daily_full_scan_hour,
            minute=self._settings.scheduler_daily_full_scan_minute,
            second=0,
            microsecond=0,
        )
        if candidate <= self._last_full_scan_at:
            candidate += timedelta(days=1)
        return candidate if candidate >= now else candidate + timedelta(days=1)

    def _is_semantic_maintenance_due(self, now: datetime) -> bool:
        next_semantic_run = self._next_semantic_maintenance_at(now)
        return next_semantic_run is None or now >= next_semantic_run

    def _next_semantic_maintenance_at(self, now: datetime) -> datetime | None:
        if self._last_semantic_maintenance_at is None:
            return now
        return self._last_semantic_maintenance_at + timedelta(
            seconds=max(1, self._settings.semantic_scheduler_interval_seconds)
        )
