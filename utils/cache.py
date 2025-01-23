from typing import TypeVar, Generic, Optional, Any, Dict
from dataclasses import dataclass
import time
import asyncio
import json
from pathlib import Path
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry(Generic[T]):
    value: T
    timestamp: float
    expires: float
    metadata: Dict[str, Any]
    access_count: int = 0

class PersistentCache:
    def __init__(self, cache_dir: str, ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.lock = asyncio.Lock()
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    async def get(self, key: str) -> Optional[Any]:
        try:
            async with self.lock:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    if time.time() < entry.expires:
                        entry.access_count += 1
                        self.metrics['hits'] += 1
                        return entry.value
                    del self.memory_cache[key]
                    self.metrics['evictions'] += 1

                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    try:
                        import gzip
                        import pickle
                        with gzip.open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                            if time.time() < data['expires']:
                                entry = CacheEntry(
                                    value=data['value'],
                                    timestamp=data['timestamp'],
                                    expires=data['expires'],
                                    metadata=data['metadata']
                                )
                                self.memory_cache[key] = entry
                                self.metrics['hits'] += 1
                                return entry.value
                            cache_file.unlink()
                            self.metrics['evictions'] += 1
                    except Exception:
                        if cache_file.exists():
                            cache_file.unlink()
                self.metrics['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
            return None

    async def set(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        async with self.lock:
            now = time.time()
            entry = CacheEntry(
                value=value,
                timestamp=now,
                expires=now + self.ttl,
                metadata=metadata or {}
            )
            self.memory_cache[key] = entry
            
            cache_file = self.cache_dir / f"{key}.cache"
            import gzip
            import pickle
            try:
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'timestamp': entry.timestamp,
                        'expires': entry.expires,
                        'metadata': entry.metadata
                    }, f)
            except Exception as e:
                logger.error(f"Cache write error: {str(e)}")

    async def clear(self, older_than: Optional[int] = None) -> int:
        cleared = 0
        now = time.time()
        async with self.lock:
            expired = [k for k, v in self.memory_cache.items() 
                      if now > v.expires or (older_than and now - v.timestamp > older_than)]
            for k in expired:
                del self.memory_cache[k]
                cleared += 1
                self.metrics['evictions'] += 1

            for cache_file in self.cache_dir.glob('*.cache'):
                try:
                    data = json.loads(cache_file.read_text())
                    if now > data['expires'] or (older_than and now - data['timestamp'] > older_than):
                        cache_file.unlink()
                        cleared += 1
                        self.metrics['evictions'] += 1
                except Exception:
                    cache_file.unlink()
                    cleared += 1
                    self.metrics['evictions'] += 1

        return cleared

    async def get_metrics(self) -> Dict[str, int]:
        async with self.lock:
            return self.metrics.copy()

    async def optimize(self) -> None:
        async with self.lock:
            await self.clear()
            
            self.memory_cache = {k: v for k, v in self.memory_cache.items()}
            
            temp_dir = self.cache_dir / 'temp'
            temp_dir.mkdir(exist_ok=True)
            try:
                for cache_file in self.cache_dir.glob('*.cache'):
                    if cache_file.stat().st_size > 0:
                        import shutil
                        shutil.copy2(cache_file, temp_dir / cache_file.name)
                
                for file in self.cache_dir.glob('*.cache'):
                    file.unlink()
                
                for file in temp_dir.glob('*.cache'):
                    file.rename(self.cache_dir / file.name)
                    
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
