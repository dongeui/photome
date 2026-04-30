"""Local scheduler primitives for serialized Phase 1/2 scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from typing import Optional

from app.core.settings import AppSettings
from app.models.runtime import SchedulerRuntimeConfig
from app.services.processing.pipeline import ProcessingPipeline
from app.services.processing.pipeline import LibraryJobBusyError
from sqlalchemy.orm import Session, sessionmaker


logger = logging.getLogger(__name__)
SCHEDULE_OPTIONS_HOURS: tuple[int | None, ...] = (None, 6, 12, 24)


@dataclass(frozen=True)
class SchedulerSnapshot:
    enabled: bool
    running: bool
    poll_interval_seconds: int
    daily_full_scan_hour: int
    daily_full_scan_minute: int
    semantic_scheduler_enabled: bool
    semantic_scheduler_interval_seconds: int
    phase1_interval_hours: int | None
    phase2_interval_hours: int | None
    last_poll_at: Optional[datetime]
    last_full_scan_at: Optional[datetime]
    next_poll_at: Optional[datetime]
    next_full_scan_at: Optional[datetime]
    last_semantic_maintenance_at: Optional[datetime]
    next_semantic_maintenance_at: Optional[datetime]


class SchedulerService:
    def __init__(self, settings: AppSettings, pipeline: ProcessingPipeline, session_factory: sessionmaker[Session]) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._session_factory = session_factory
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        config = self._load_runtime_config()
        return bool(config.phase1_interval_hours or config.phase2_interval_hours)

    def start(self) -> None:
        if self._running:
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
        config = self._load_runtime_config()
        if config.phase1_interval_hours is not None and self._is_phase1_due(config, now):
            try:
                self._pipeline.submit_scan_job(full_scan=True, run_now=True, trigger="scheduler-phase1")
                self._set_last_phase1_run(now)
            except LibraryJobBusyError:
                logger.debug("phase 1 scheduler skipped: library job already active")
            except Exception:
                logger.exception("phase 1 scheduler tick failed")
        if config.phase2_interval_hours is not None and self._is_phase2_due(config, now):
            try:
                self._pipeline.submit_semantic_maintenance_job(batch_size=100, run_now=True, trigger="scheduler-phase2")
                self._set_last_phase2_run(now)
            except LibraryJobBusyError:
                logger.debug("phase 2 scheduler skipped: library job already active")
            except Exception:
                logger.exception("phase 2 scheduler tick failed")
        return self.snapshot(now)

    def snapshot(self, now: datetime | None = None) -> SchedulerSnapshot:
        now = now or datetime.utcnow()
        config = self._load_runtime_config()
        return SchedulerSnapshot(
            enabled=bool(config.phase1_interval_hours or config.phase2_interval_hours),
            running=self._running,
            poll_interval_seconds=self._settings.scheduler_poll_interval_seconds,
            daily_full_scan_hour=self._settings.scheduler_daily_full_scan_hour,
            daily_full_scan_minute=self._settings.scheduler_daily_full_scan_minute,
            semantic_scheduler_enabled=self._settings.semantic_scheduler_enabled,
            semantic_scheduler_interval_seconds=self._settings.semantic_scheduler_interval_seconds,
            phase1_interval_hours=config.phase1_interval_hours,
            phase2_interval_hours=config.phase2_interval_hours,
            last_poll_at=None,
            last_full_scan_at=config.last_phase1_run_at,
            next_poll_at=None,
            next_full_scan_at=self._next_phase1_run_at(config, now),
            last_semantic_maintenance_at=config.last_phase2_run_at,
            next_semantic_maintenance_at=self._next_phase2_run_at(config, now),
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("scheduler tick failed")
            self._stop_event.wait(1.0)

    def cycle_phase_schedule(self, phase: str) -> SchedulerSnapshot:
        now = datetime.utcnow()
        with self._session_factory() as session:
            config = self._ensure_runtime_config_row(session)
            current = config.phase1_interval_hours if phase == "phase1" else config.phase2_interval_hours
            next_value = self._cycle_value(current)
            if phase == "phase1":
                config.phase1_interval_hours = next_value
                if current is None and next_value is not None:
                    config.last_phase1_run_at = now
            elif phase == "phase2":
                config.phase2_interval_hours = next_value
                if current is None and next_value is not None:
                    config.last_phase2_run_at = now
            else:
                raise ValueError(f"Unknown phase: {phase}")
            session.commit()
        return self.snapshot()

    def _cycle_value(self, current: int | None) -> int | None:
        index = SCHEDULE_OPTIONS_HOURS.index(current) if current in SCHEDULE_OPTIONS_HOURS else 0
        return SCHEDULE_OPTIONS_HOURS[(index + 1) % len(SCHEDULE_OPTIONS_HOURS)]

    def _load_runtime_config(self) -> SchedulerRuntimeConfig:
        with self._session_factory() as session:
            config = self._ensure_runtime_config_row(session)
            if session.new:
                session.commit()
                session.refresh(config)
            session.expunge(config)
            return config

    def _ensure_runtime_config_row(self, session: Session) -> SchedulerRuntimeConfig:
        config = session.get(SchedulerRuntimeConfig, 1)
        if config is None:
            config = SchedulerRuntimeConfig(id=1)
            session.add(config)
            session.flush()
        return config

    def _is_phase1_due(self, config: SchedulerRuntimeConfig, now: datetime) -> bool:
        next_run = self._next_phase1_run_at(config, now)
        return next_run is not None and now >= next_run

    def _is_phase2_due(self, config: SchedulerRuntimeConfig, now: datetime) -> bool:
        next_run = self._next_phase2_run_at(config, now)
        return next_run is not None and now >= next_run

    def _next_phase1_run_at(self, config: SchedulerRuntimeConfig, now: datetime) -> datetime | None:
        if config.phase1_interval_hours is None:
            return None
        if config.last_phase1_run_at is None:
            return now
        return config.last_phase1_run_at + timedelta(hours=max(1, config.phase1_interval_hours))

    def _next_phase2_run_at(self, config: SchedulerRuntimeConfig, now: datetime) -> datetime | None:
        if config.phase2_interval_hours is None:
            return None
        if config.last_phase2_run_at is None:
            return now
        return config.last_phase2_run_at + timedelta(hours=max(1, config.phase2_interval_hours))

    def _set_last_phase1_run(self, now: datetime) -> None:
        with self._session_factory() as session:
            config = self._ensure_runtime_config_row(session)
            config.last_phase1_run_at = now
            session.commit()

    def _set_last_phase2_run(self, now: datetime) -> None:
        with self._session_factory() as session:
            config = self._ensure_runtime_config_row(session)
            config.last_phase2_run_at = now
            session.commit()
