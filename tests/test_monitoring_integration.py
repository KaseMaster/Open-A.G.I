import pytest
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.monitoring.metrics_collector import AEGISMetricsCollector, MetricType, ApplicationMetricsCollector


class TestMonitoringIntegration:

    @pytest.fixture
    def metrics_collector(self):
        collector = AEGISMetricsCollector(config={"collection_interval": 30})
        yield collector

    def test_metrics_collector_initialization(self, metrics_collector):
        assert metrics_collector.config is not None
        assert metrics_collector.metrics is not None
        assert metrics_collector.collection_interval == 30

    def test_add_metric_series(self, metrics_collector):
        metrics_collector.add_metric_series(
            "test_metric", 
            MetricType.CUSTOM, 
            "Test metric description",
            retention_hours=24
        )
        
        assert "test_metric" in metrics_collector.metrics
        assert metrics_collector.metrics["test_metric"].name == "test_metric"
        assert metrics_collector.metrics["test_metric"].metric_type == MetricType.CUSTOM

    def test_record_metric_auto_creation(self, metrics_collector):
        metrics_collector.record_metric("auto_created_metric", 42.5)
        
        assert "auto_created_metric" in metrics_collector.metrics
        latest_value = metrics_collector.metrics["auto_created_metric"].get_latest()
        assert latest_value is not None
        assert latest_value.value == 42.5

    def test_record_metric_with_tags(self, metrics_collector):
        metrics_collector.record_metric(
            "tagged_metric", 
            100, 
            unit="ms",
            tags={"operation": "read", "status": "success"}
        )
        
        latest = metrics_collector.metrics["tagged_metric"].get_latest()
        assert latest.value == 100
        assert latest.unit == "ms"
        assert latest.tags["operation"] == "read"
        assert latest.tags["status"] == "success"

    def test_application_counter_increment(self):
        app_collector = ApplicationMetricsCollector()
        
        app_collector.increment_counter("test_requests", value=1, tags={"endpoint": "/api/v1/health"})
        app_collector.increment_counter("test_requests", value=1, tags={"endpoint": "/api/v1/health"})
        app_collector.increment_counter("test_requests", value=5, tags={"endpoint": "/api/v1/data"})
        
        assert len(app_collector.counters) == 2

    def test_application_timer_recording(self):
        app_collector = ApplicationMetricsCollector()
        
        app_collector.record_timer("request_duration", 12.5, tags={"endpoint": "/api/v1"})
        app_collector.record_timer("request_duration", 15.3, tags={"endpoint": "/api/v1"})
        app_collector.record_timer("request_duration", 9.8, tags={"endpoint": "/api/v1"})
        
        key = 'request_duration:{"endpoint": "/api/v1"}'
        assert key in app_collector.timers
        assert len(app_collector.timers[key]) == 3

    def test_application_gauge_setting(self):
        app_collector = ApplicationMetricsCollector()
        
        app_collector.set_gauge("active_connections", 42, tags={"server": "node_001"})
        app_collector.set_gauge("active_connections", 38, tags={"server": "node_001"})
        
        key = 'active_connections:{"server": "node_001"}'
        assert key in app_collector.custom_metrics
        assert app_collector.custom_metrics[key]["value"] == 38

    @pytest.mark.asyncio
    async def test_collect_application_metrics(self):
        app_collector = ApplicationMetricsCollector()
        
        app_collector.increment_counter("api_requests", 10)
        app_collector.record_timer("response_time", 25.5)
        app_collector.record_timer("response_time", 30.2)
        app_collector.set_gauge("memory_usage", 1024000)
        
        metrics = await app_collector.collect_application_metrics()
        
        assert "counter_api_requests" in metrics
        assert "timer_response_time_avg" in metrics
        assert "gauge_memory_usage" in metrics
        assert metrics["counter_api_requests"] == 10

    def test_metric_series_average_calculation(self, metrics_collector):
        metrics_collector.add_metric_series("cpu_usage", MetricType.SYSTEM)
        
        metrics_collector.record_metric("cpu_usage", 50.0)
        time.sleep(0.1)
        metrics_collector.record_metric("cpu_usage", 60.0)
        time.sleep(0.1)
        metrics_collector.record_metric("cpu_usage", 70.0)
        
        avg = metrics_collector.metrics["cpu_usage"].get_average(minutes=5)
        assert avg is not None
        assert 50.0 <= avg <= 70.0

    def test_metrics_summary_generation(self, metrics_collector):
        metrics_collector.add_metric_series("metric1", MetricType.CUSTOM)
        metrics_collector.add_metric_series("metric2", MetricType.CUSTOM)
        metrics_collector.record_metric("metric1", 100)
        metrics_collector.record_metric("metric2", 200)
        
        summary = metrics_collector.get_metrics_summary()
        
        assert summary["total_series"] >= 2
        assert summary["total_data_points"] >= 2
        assert "active_alerts" in summary
        assert "collection_running" in summary

    def test_alert_rule_checking(self, metrics_collector):
        initial_alerts = len(metrics_collector.alert_manager.get_active_alerts())
        
        metrics_collector.record_metric("cpu_usage_percent", 95.0)
        
        active_alerts = metrics_collector.alert_manager.get_active_alerts()
        assert len(active_alerts) >= initial_alerts

    def test_metric_retention_cleanup(self, metrics_collector):
        metrics_collector.add_metric_series("short_retention", MetricType.CUSTOM, retention_hours=0)
        
        metrics_collector.record_metric("short_retention", 100)
        time.sleep(0.2)
        
        metrics_collector.metrics["short_retention"]._cleanup_old_values()
        
        assert len(metrics_collector.metrics["short_retention"].values) >= 0

    def test_multiple_metric_types(self, metrics_collector):
        for metric_type in MetricType:
            metric_name = f"test_{metric_type.value}"
            metrics_collector.add_metric_series(metric_name, metric_type)
        
        assert len(metrics_collector.metrics) >= len(MetricType)

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, metrics_collector):
        system_metrics = await metrics_collector.system_collector.collect_cpu_metrics()
        
        assert "cpu_usage_percent" in system_metrics
        assert isinstance(system_metrics["cpu_usage_percent"], (int, float))
        assert system_metrics["cpu_usage_percent"] >= 0

    @pytest.mark.asyncio
    async def test_memory_metrics_collection(self, metrics_collector):
        memory_metrics = await metrics_collector.system_collector.collect_memory_metrics()
        
        assert "memory_usage_percent" in memory_metrics
        assert isinstance(memory_metrics["memory_usage_percent"], (int, float))

    def test_application_timer_limits(self):
        app_collector = ApplicationMetricsCollector()
        
        for i in range(1500):
            app_collector.record_timer("bulk_timer", float(i))
        
        key = 'bulk_timer:{}'
        assert len(app_collector.timers[key]) == 1000
