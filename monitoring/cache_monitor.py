"""
Cache Performance Monitor for AURA AI Trading System.
Monitors cache hit/miss rates and provides performance insights.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CachePerformanceSnapshot:
    """Snapshot of cache performance at a point in time."""
    timestamp: datetime
    cache_name: str
    hits: int
    misses: int
    hit_rate: float
    cache_size: int
    max_size: int
    evictions: int = 0
    avg_lookup_time_ms: float = 0.0


class CachePerformanceMonitor:
    """Monitors and tracks cache performance metrics."""
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: List[CachePerformanceSnapshot] = []
        self.lookup_times: Dict[str, List[float]] = {}
        self._lock = Lock()
        
        # Performance thresholds
        self.min_hit_rate_threshold = 0.7  # 70% minimum hit rate
        self.max_lookup_time_ms = 10.0     # 10ms maximum lookup time
        
        logger.info("CachePerformanceMonitor initialized")
    
    def record_lookup_time(self, cache_name: str, lookup_time_ms: float):
        """Record a cache lookup time."""
        with self._lock:
            if cache_name not in self.lookup_times:
                self.lookup_times[cache_name] = []
            
            self.lookup_times[cache_name].append(lookup_time_ms)
            
            # Keep only recent lookup times (last 100)
            if len(self.lookup_times[cache_name]) > 100:
                self.lookup_times[cache_name] = self.lookup_times[cache_name][-100:]
    
    def take_snapshot(self, cache_stats: Dict[str, Any], cache_name: str = "default"):
        """Take a performance snapshot."""
        with self._lock:
            # Calculate average lookup time
            avg_lookup_time = 0.0
            if cache_name in self.lookup_times and self.lookup_times[cache_name]:
                avg_lookup_time = sum(self.lookup_times[cache_name]) / len(self.lookup_times[cache_name])
            
            snapshot = CachePerformanceSnapshot(
                timestamp=datetime.now(),
                cache_name=cache_name,
                hits=cache_stats.get('hits', 0),
                misses=cache_stats.get('misses', 0),
                hit_rate=cache_stats.get('hit_rate', 0.0),
                cache_size=cache_stats.get('current_size', 0),
                max_size=cache_stats.get('max_size', 0),
                avg_lookup_time_ms=avg_lookup_time
            )
            
            self.snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
            
            if not recent_snapshots:
                return {"error": "No snapshots available for the specified time window"}
            
            # Group by cache name
            cache_summaries = {}
            for snapshot in recent_snapshots:
                cache_name = snapshot.cache_name
                if cache_name not in cache_summaries:
                    cache_summaries[cache_name] = []
                cache_summaries[cache_name].append(snapshot)
            
            # Calculate summaries for each cache
            summary = {}
            for cache_name, snapshots in cache_summaries.items():
                latest = snapshots[-1]
                oldest = snapshots[0]
                
                # Calculate rates over the time window
                total_requests_delta = (latest.hits + latest.misses) - (oldest.hits + oldest.misses)
                hit_rate_avg = sum(s.hit_rate for s in snapshots) / len(snapshots)
                lookup_time_avg = sum(s.avg_lookup_time_ms for s in snapshots) / len(snapshots)
                
                summary[cache_name] = {
                    'current_hit_rate': latest.hit_rate,
                    'average_hit_rate': hit_rate_avg,
                    'current_cache_size': latest.cache_size,
                    'max_cache_size': latest.max_size,
                    'cache_utilization': latest.cache_size / latest.max_size if latest.max_size > 0 else 0,
                    'total_requests_in_window': total_requests_delta,
                    'average_lookup_time_ms': lookup_time_avg,
                    'performance_issues': self._identify_performance_issues(snapshots)
                }
            
            return {
                'time_window_minutes': time_window_minutes,
                'snapshot_count': len(recent_snapshots),
                'cache_summaries': summary,
                'overall_health': self._calculate_overall_health(summary)
            }
    
    def _identify_performance_issues(self, snapshots: List[CachePerformanceSnapshot]) -> List[str]:
        """Identify performance issues from snapshots."""
        issues = []
        
        if not snapshots:
            return issues
        
        latest = snapshots[-1]
        
        # Check hit rate
        if latest.hit_rate < self.min_hit_rate_threshold:
            issues.append(f"Low hit rate: {latest.hit_rate:.2%} (threshold: {self.min_hit_rate_threshold:.2%})")
        
        # Check lookup time
        if latest.avg_lookup_time_ms > self.max_lookup_time_ms:
            issues.append(f"High lookup time: {latest.avg_lookup_time_ms:.2f}ms (threshold: {self.max_lookup_time_ms}ms)")
        
        # Check cache utilization
        utilization = latest.cache_size / latest.max_size if latest.max_size > 0 else 0
        if utilization > 0.9:
            issues.append(f"High cache utilization: {utilization:.2%}")
        
        # Check for declining hit rate trend
        if len(snapshots) >= 5:
            recent_hit_rates = [s.hit_rate for s in snapshots[-5:]]
            if len(recent_hit_rates) >= 2:
                trend = (recent_hit_rates[-1] - recent_hit_rates[0]) / len(recent_hit_rates)
                if trend < -0.05:  # Declining by more than 5% per snapshot
                    issues.append("Declining hit rate trend detected")
        
        return issues
    
    def _calculate_overall_health(self, cache_summaries: Dict[str, Any]) -> str:
        """Calculate overall cache health score."""
        if not cache_summaries:
            return "unknown"
        
        total_issues = sum(len(summary.get('performance_issues', [])) for summary in cache_summaries.values())
        avg_hit_rate = sum(summary.get('current_hit_rate', 0) for summary in cache_summaries.values()) / len(cache_summaries)
        
        if total_issues == 0 and avg_hit_rate >= 0.8:
            return "excellent"
        elif total_issues <= 1 and avg_hit_rate >= 0.7:
            return "good"
        elif total_issues <= 3 and avg_hit_rate >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def get_cache_recommendations(self) -> List[str]:
        """Get recommendations for cache optimization."""
        recommendations = []
        
        summary = self.get_performance_summary(60)  # Last hour
        cache_summaries = summary.get('cache_summaries', {})
        
        for cache_name, cache_summary in cache_summaries.items():
            hit_rate = cache_summary.get('current_hit_rate', 0)
            utilization = cache_summary.get('cache_utilization', 0)
            lookup_time = cache_summary.get('average_lookup_time_ms', 0)
            
            # Hit rate recommendations
            if hit_rate < 0.5:
                recommendations.append(f"{cache_name}: Consider increasing cache size or reviewing cache key strategy")
            elif hit_rate < 0.7:
                recommendations.append(f"{cache_name}: Hit rate could be improved - review data access patterns")
            
            # Utilization recommendations
            if utilization > 0.95:
                recommendations.append(f"{cache_name}: Cache is nearly full - consider increasing max size")
            elif utilization < 0.3:
                recommendations.append(f"{cache_name}: Cache is underutilized - consider reducing max size")
            
            # Performance recommendations
            if lookup_time > 5.0:
                recommendations.append(f"{cache_name}: High lookup times - consider optimizing cache implementation")
        
        if not recommendations:
            recommendations.append("Cache performance is optimal - no recommendations at this time")
        
        return recommendations
    
    def export_metrics(self, filepath: str):
        """Export cache metrics to JSON file."""
        with self._lock:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'snapshots': [asdict(snapshot) for snapshot in self.snapshots],
                'performance_summary': self.get_performance_summary(60),
                'recommendations': self.get_cache_recommendations()
            }
            
            # Convert datetime objects to strings for JSON serialization
            for snapshot_data in data['snapshots']:
                if isinstance(snapshot_data['timestamp'], datetime):
                    snapshot_data['timestamp'] = snapshot_data['timestamp'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Cache metrics exported to {filepath}")
    
    def clear_history(self):
        """Clear all performance history."""
        with self._lock:
            self.snapshots.clear()
            self.lookup_times.clear()
            logger.info("Cache performance history cleared")


class CacheLookupTimer:
    """Context manager for timing cache lookups."""
    
    def __init__(self, monitor: CachePerformanceMonitor, cache_name: str):
        self.monitor = monitor
        self.cache_name = cache_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.monitor.record_lookup_time(self.cache_name, elapsed_ms)


# Global monitor instance
_cache_monitor = None


def get_cache_monitor() -> CachePerformanceMonitor:
    """Get or create the global cache monitor instance."""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CachePerformanceMonitor()
    return _cache_monitor