"""Filesystem scanner for source roots."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Iterator

from app.core.contracts import FileScanRecord, MediaKind, media_kind_from_path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScannerConfig:
    source_roots: tuple[Path, ...]
    include_hidden_files: bool = False
    follow_symlinks: bool = False
    stability_window_seconds: int = 300


class ScannerService:
    def __init__(self, config: ScannerConfig) -> None:
        self._config = config

    @property
    def config(self) -> ScannerConfig:
        return self._config

    def iter_files(self) -> Iterator[FileScanRecord]:
        for source_root in self._config.source_roots:
            yield from self._iter_root(source_root)

    def _iter_root(self, source_root: Path) -> Iterator[FileScanRecord]:
        if not source_root.exists():
            logger.warning("source root missing", extra={"source_root": str(source_root)})
            return

        for current_path in self._walk(source_root):
            try:
                stat_result = os.stat(current_path, follow_symlinks=self._config.follow_symlinks)
            except OSError as exc:
                logger.warning(
                    "unable to stat file",
                    extra={"path": str(current_path), "error": str(exc)},
                )
                continue

            relative_path = current_path.relative_to(source_root)
            yield FileScanRecord(
                source_root=source_root,
                path=current_path,
                relative_path=relative_path,
                size_bytes=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                media_kind=media_kind_from_path(current_path),
            )

    def _walk(self, source_root: Path) -> Iterator[Path]:
        stack = [source_root]
        while stack:
            current_dir = stack.pop()
            try:
                with os.scandir(current_dir) as iterator:
                    entries = list(iterator)
            except OSError as exc:
                logger.warning(
                    "unable to read directory",
                    extra={"path": str(current_dir), "error": str(exc)},
                )
                continue

            for entry in sorted(entries, key=lambda item: item.name):
                if not self._config.include_hidden_files and entry.name.startswith("."):
                    continue

                path = Path(entry.path)
                try:
                    if entry.is_symlink() and not self._config.follow_symlinks:
                        continue
                    if entry.is_dir(follow_symlinks=self._config.follow_symlinks):
                        stack.append(path)
                        continue
                    if entry.is_file(follow_symlinks=self._config.follow_symlinks):
                        yield path
                except OSError as exc:
                    logger.warning(
                        "unable to inspect entry",
                        extra={"path": str(path), "error": str(exc)},
                    )
