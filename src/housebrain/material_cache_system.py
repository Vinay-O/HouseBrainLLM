"""
Material Cache System for HouseBrain Professional

This module provides intelligent caching for the material library:
- In-memory caching with LRU eviction
- Persistent disk caching for large datasets
- Smart cache invalidation and updates
- Regional material caching
- Performance monitoring and optimization
- Preloading strategies for common materials
"""

from __future__ import annotations

import json
import pickle
import hashlib
import time
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import weakref


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int
    dependencies: Set[str]
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > ttl_seconds
    
    def touch(self):
        """Update access information"""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        total_requests = self.hits + self.misses
        self.hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0


class MaterialCacheSystem:
    """Intelligent caching system for material library"""
    
    def __init__(
        self,
        max_memory_size: int = 100 * 1024 * 1024,  # 100MB
        max_entries: int = 1000,
        ttl_seconds: float = 3600,  # 1 hour
        cache_dir: str = "material_cache",
        enable_disk_cache: bool = True
    ):
        self.max_memory_size = max_memory_size
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cache_dir = Path(cache_dir)
        self.enable_disk_cache = enable_disk_cache
        
        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU eviction
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Preloading configuration
        self.preload_patterns = set()
        self.regional_cache = {}
        
        # Setup disk cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Weak reference to material library for invalidation
        self.material_library_ref = None
        
        print("💾 Material Cache System Initialized")
        print(f"   Memory limit: {self.max_memory_size / 1024 / 1024:.1f}MB")
        print(f"   Max entries: {self.max_entries}")
        print(f"   TTL: {self.ttl_seconds}s")
    
    def set_material_library(self, material_library):
        """Set reference to material library for cache invalidation"""
        self.material_library_ref = weakref.ref(material_library)
    
    def get(self, key: str, region: str = "global") -> Optional[Any]:
        """Get cached material data"""
        
        cache_key = self._create_cache_key(key, region)
        
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                
                # Check if expired
                if entry.is_expired(self.ttl_seconds):
                    self._remove_from_memory_cache(cache_key)
                else:
                    entry.touch()
                    self._update_access_order(cache_key)
                    self.stats.hits += 1
                    self.stats.update_hit_rate()
                    return entry.data
            
            # Check disk cache if enabled
            if self.enable_disk_cache:
                disk_data = self._get_from_disk_cache(cache_key)
                if disk_data is not None:
                    # Add back to memory cache
                    self._add_to_memory_cache(cache_key, disk_data, set())
                    self.stats.hits += 1
                    self.stats.update_hit_rate()
                    return disk_data
            
            # Cache miss
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None
    
    def put(
        self,
        key: str,
        data: Any,
        region: str = "global",
        dependencies: Set[str] = None
    ) -> None:
        """Cache material data"""
        
        cache_key = self._create_cache_key(key, region)
        dependencies = dependencies or set()
        
        with self.lock:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, data, dependencies)
            
            # Add to disk cache if enabled
            if self.enable_disk_cache:
                self._add_to_disk_cache(cache_key, data)
    
    def invalidate(self, key: str, region: str = "global") -> bool:
        """Invalidate specific cache entry"""
        
        cache_key = self._create_cache_key(key, region)
        
        with self.lock:
            removed = False
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                self._remove_from_memory_cache(cache_key)
                removed = True
            
            # Remove from disk cache
            if self.enable_disk_cache:
                disk_removed = self._remove_from_disk_cache(cache_key)
                removed = removed or disk_removed
            
            return removed
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all entries matching pattern"""
        
        with self.lock:
            keys_to_remove = []
            
            for key in self.memory_cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_from_memory_cache(key)
                if self.enable_disk_cache:
                    self._remove_from_disk_cache(key)
            
            return len(keys_to_remove)
    
    def invalidate_dependencies(self, dependency: str) -> int:
        """Invalidate all entries that depend on a specific dependency"""
        
        with self.lock:
            keys_to_remove = []
            
            for key, entry in self.memory_cache.items():
                if dependency in entry.dependencies:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_from_memory_cache(key)
                if self.enable_disk_cache:
                    self._remove_from_disk_cache(key)
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cached data"""
        
        with self.lock:
            self.memory_cache.clear()
            self.access_order.clear()
            self.stats = CacheStats()
            
            if self.enable_disk_cache:
                self._clear_disk_cache()
    
    def preload_materials(self, material_patterns: List[str], regions: List[str] = None) -> int:
        """Preload materials matching patterns"""
        
        if regions is None:
            regions = ["global"]
        
        preloaded_count = 0
        
        # Get material library reference
        if self.material_library_ref and self.material_library_ref():
            material_lib = self.material_library_ref()
            
            for region in regions:
                for pattern in material_patterns:
                    # Get materials matching pattern
                    materials = self._get_materials_by_pattern(material_lib, pattern, region)
                    
                    for material_name, material_data in materials.items():
                        cache_key = self._create_cache_key(material_name, region)
                        
                        if cache_key not in self.memory_cache:
                            self.put(material_name, material_data, region)
                            preloaded_count += 1
        
        print(f"💾 Preloaded {preloaded_count} materials")
        return preloaded_count
    
    def preload_regional_materials(self, region: str) -> int:
        """Preload all materials for a specific region"""
        
        preloaded_count = 0
        
        if self.material_library_ref and self.material_library_ref():
            material_lib = self.material_library_ref()
            
            try:
                regional_materials = material_lib.get_regional_materials(region)
                
                for material_name, material_data in regional_materials.items():
                    self.put(material_name, material_data, region)
                    preloaded_count += 1
                
            except AttributeError:
                # Material library doesn't have get_regional_materials method
                pass
        
        print(f"💾 Preloaded {preloaded_count} materials for region: {region}")
        return preloaded_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        with self.lock:
            self.stats.entry_count = len(self.memory_cache)
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self.memory_cache.values())
            
            return {
                "memory_cache": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "hit_rate": f"{self.stats.hit_rate:.1f}%",
                    "entry_count": self.stats.entry_count,
                    "total_size_mb": self.stats.total_size_bytes / 1024 / 1024,
                    "max_size_mb": self.max_memory_size / 1024 / 1024,
                    "utilization": f"{(self.stats.total_size_bytes / self.max_memory_size * 100):.1f}%",
                    "evictions": self.stats.evictions
                },
                "disk_cache": {
                    "enabled": self.enable_disk_cache,
                    "directory": str(self.cache_dir),
                    "file_count": len(list(self.cache_dir.glob("*.cache"))) if self.enable_disk_cache else 0
                },
                "performance": {
                    "average_entry_size_kb": (self.stats.total_size_bytes / self.stats.entry_count / 1024) if self.stats.entry_count > 0 else 0,
                    "most_accessed": self._get_most_accessed_entries(5),
                    "cache_efficiency": "high" if self.stats.hit_rate > 80 else "medium" if self.stats.hit_rate > 60 else "low"
                }
            }
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance and cleanup"""
        
        optimization_result = {
            "expired_entries_removed": 0,
            "entries_compressed": 0,
            "disk_cache_cleaned": 0,
            "memory_freed_mb": 0.0
        }
        
        with self.lock:
            initial_size = self.stats.total_size_bytes
            
            # Remove expired entries
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if entry.is_expired(self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_memory_cache(key)
                optimization_result["expired_entries_removed"] += 1
            
            # Clean disk cache
            if self.enable_disk_cache:
                optimization_result["disk_cache_cleaned"] = self._cleanup_disk_cache()
            
            final_size = sum(entry.size_bytes for entry in self.memory_cache.values())
            optimization_result["memory_freed_mb"] = (initial_size - final_size) / 1024 / 1024
        
        print(f"💾 Cache optimized: {optimization_result}")
        return optimization_result
    
    def cached_method(self, cache_key_func: Optional[Callable] = None, ttl: Optional[float] = None):
        """Decorator for caching method results"""
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_method_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    # Private methods
    
    def _create_cache_key(self, key: str, region: str) -> str:
        """Create cache key with region prefix"""
        return f"{region}:{key}"
    
    def _add_to_memory_cache(self, key: str, data: Any, dependencies: Set[str]) -> None:
        """Add entry to memory cache"""
        
        # Calculate data size
        data_size = self._calculate_size(data)
        
        # Check if we need to evict entries
        self._ensure_cache_capacity(data_size)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time(),
            size_bytes=data_size,
            dependencies=dependencies
        )
        
        # Add to cache
        self.memory_cache[key] = entry
        self.access_order.append(key)
        
        # Update statistics
        self.stats.total_size_bytes += data_size
        self.stats.entry_count = len(self.memory_cache)
    
    def _remove_from_memory_cache(self, key: str) -> None:
        """Remove entry from memory cache"""
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            
            del self.memory_cache[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            
            self.stats.entry_count = len(self.memory_cache)
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order"""
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _ensure_cache_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry"""
        
        # Check size limit
        while (self.stats.total_size_bytes + new_entry_size > self.max_memory_size and
               self.access_order):
            oldest_key = self.access_order[0]
            self._remove_from_memory_cache(oldest_key)
            self.stats.evictions += 1
        
        # Check entry count limit
        while len(self.memory_cache) >= self.max_entries and self.access_order:
            oldest_key = self.access_order[0]
            self._remove_from_memory_cache(oldest_key)
            self.stats.evictions += 1
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        
        try:
            # Use pickle to estimate size
            return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, dict):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return 1000  # Default 1KB estimate
    
    def _get_from_disk_cache(self, key: str) -> Optional[Any]:
        """Get data from disk cache"""
        
        if not self.enable_disk_cache:
            return None
        
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if expired
                if time.time() - cached_data['timestamp'] > self.ttl_seconds:
                    cache_file.unlink()  # Remove expired file
                    return None
                
                return cached_data['data']
        except Exception:
            # If there's any error reading the cache file, remove it
            try:
                cache_file.unlink()
            except Exception:
                pass
        
        return None
    
    def _add_to_disk_cache(self, key: str, data: Any) -> None:
        """Add data to disk cache"""
        
        if not self.enable_disk_cache:
            return
        
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        try:
            cached_data = {
                'key': key,
                'data': data,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"Warning: Failed to write disk cache: {e}")
    
    def _remove_from_disk_cache(self, key: str) -> bool:
        """Remove data from disk cache"""
        
        if not self.enable_disk_cache:
            return False
        
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        try:
            if cache_file.exists():
                cache_file.unlink()
                return True
        except Exception:
            pass
        
        return False
    
    def _clear_disk_cache(self) -> None:
        """Clear all disk cache files"""
        
        if not self.enable_disk_cache:
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear disk cache: {e}")
    
    def _cleanup_disk_cache(self) -> int:
        """Clean up expired disk cache files"""
        
        if not self.enable_disk_cache:
            return 0
        
        cleaned_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if time.time() - cached_data['timestamp'] > self.ttl_seconds:
                        cache_file.unlink()
                        cleaned_count += 1
                except Exception:
                    # If file is corrupted, remove it
                    cache_file.unlink()
                    cleaned_count += 1
        except Exception:
            pass
        
        return cleaned_count
    
    def _hash_key(self, key: str) -> str:
        """Create hash of cache key for filename"""
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _get_materials_by_pattern(
        self,
        material_lib,
        pattern: str,
        region: str
    ) -> Dict[str, Any]:
        """Get materials matching pattern from library"""
        
        materials = {}
        
        try:
            # Try to get all materials by category
            if hasattr(material_lib, 'get_materials_by_category'):
                for category in ['concrete', 'timber', 'metals', 'glass', 'stone']:
                    if pattern.lower() in category.lower():
                        category_materials = material_lib.get_materials_by_category(category)
                        materials.update(category_materials)
            
            # Try to get specific material
            if hasattr(material_lib, 'get_material'):
                try:
                    material_data = material_lib.get_material(pattern, region)
                    if material_data:
                        materials[pattern] = material_data
                except Exception:
                    pass
        
        except Exception:
            pass
        
        return materials
    
    def _get_most_accessed_entries(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get most accessed cache entries"""
        
        entries = list(self.memory_cache.values())
        entries.sort(key=lambda x: x.access_count, reverse=True)
        
        return [
            {
                "key": entry.key,
                "access_count": entry.access_count,
                "size_kb": entry.size_bytes / 1024
            }
            for entry in entries[:count]
        ]
    
    def _generate_method_cache_key(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Generate cache key for method call"""
        
        # Create a deterministic key from method name and arguments
        key_parts = [method_name]
        
        # Add args (skip self if present)
        for arg in args[1:] if args and hasattr(args[0], method_name) else args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(type(arg).__name__))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={type(v).__name__}")
        
        return ":".join(key_parts)


# Caching decorators for material library methods
def cached_material_method(cache_system: MaterialCacheSystem, ttl: Optional[float] = None):
    """Decorator for caching material library method results"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from method and arguments
            cache_key = f"method:{func.__name__}:{hash(str(args[1:]) + str(kwargs))}"
            
            # Check cache first
            cached_result = cache_system.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute method and cache result
            result = func(*args, **kwargs)
            cache_system.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def create_material_cache_system(
    max_memory_mb: int = 100,
    max_entries: int = 1000,
    ttl_hours: float = 1.0,
    cache_dir: str = "material_cache",
    enable_disk_cache: bool = True
) -> MaterialCacheSystem:
    """Create material cache system instance"""
    
    return MaterialCacheSystem(
        max_memory_size=max_memory_mb * 1024 * 1024,
        max_entries=max_entries,
        ttl_seconds=ttl_hours * 3600,
        cache_dir=cache_dir,
        enable_disk_cache=enable_disk_cache
    )


if __name__ == "__main__":
    # Test material cache system
    cache_system = create_material_cache_system(
        max_memory_mb=50,
        max_entries=500,
        ttl_hours=0.5,
        enable_disk_cache=True
    )
    
    print("💾 Material Cache System Test")
    print("=" * 50)
    
    # Test basic caching
    test_material = {
        "name": "Test Concrete",
        "properties": {"density": 2400, "strength": 30},
        "sustainability": {"rating": "B", "embodied_carbon": "medium"}
    }
    
    # Cache material
    cache_system.put("test_concrete", test_material, "temperate")
    print("✅ Material cached")
    
    # Retrieve material
    cached_material = cache_system.get("test_concrete", "temperate")
    if cached_material:
        print(f"✅ Material retrieved: {cached_material['name']}")
    
    # Test cache statistics
    stats = cache_system.get_cache_stats()
    print(f"Cache hit rate: {stats['memory_cache']['hit_rate']}")
    print(f"Cache entries: {stats['memory_cache']['entry_count']}")
    
    # Test cache optimization
    optimization_result = cache_system.optimize_cache()
    print(f"Optimization completed: {optimization_result}")
    
    print("\n✅ Material Cache System initialized successfully!")