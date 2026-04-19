from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """
    Cache simple en memoria con expiracion por TTL.

    La cache vive por proceso de FastAPI. Se invalida automaticamente cuando
    vence el TTL o cuando el proceso se reinicia.
    """

    def __init__(self, ttl_seconds: int, maxsize: int = 128) -> None:
        self.ttl_seconds = ttl_seconds
        self.maxsize = maxsize
        self._items: dict[Hashable, CacheEntry[T]] = {}
        self._lock = RLock()

    def get(self, key: Hashable) -> T | None:
        now = monotonic()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None

            if entry.expires_at <= now:
                self._items.pop(key, None)
                return None

            return entry.value

    def set(self, key: Hashable, value: T) -> T:
        now = monotonic()
        with self._lock:
            self._prune_expired(now)
            if len(self._items) >= self.maxsize:
                oldest_key = next(iter(self._items))
                self._items.pop(oldest_key, None)

            self._items[key] = CacheEntry(
                value=value,
                expires_at=now + self.ttl_seconds,
            )
            return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _prune_expired(self, now: float) -> None:
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
