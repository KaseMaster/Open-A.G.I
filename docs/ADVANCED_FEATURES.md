# AEGIS Framework - Advanced Features Documentation

## Table of Contents
1. [Advanced Consensus](#advanced-consensus)
2. [Error Handling and Recovery](#error-handling-and-recovery)
3. [Consensus Visualization](#consensus-visualization)
4. [Performance Optimization](#performance-optimization)
5. [Security Integration](#security-integration)

---

## Advanced Consensus

### Dynamic Validator Selection

The AEGIS Framework implements an advanced dynamic validator selection system that chooses consensus participants based on real-time performance metrics and reputation scores.

#### Key Features

1. **Performance-Based Selection**
   - Response time scoring (faster validators get higher scores)
   - Success rate tracking (95%+ success rate preferred)
   - Reliability scoring (consistent performance rewarded)
   - Recent participation boosting (fair rotation)

2. **Scoring Algorithm**
   ```python
   score = (
       response_time_score * 0.3 +
       success_rate_score * 0.3 +
       reliability_score * 0.2 +
       recency_score * 0.2
   )
   ```

3. **Usage Example**
   ```python
   from aegis.blockchain.advanced_consensus import AdvancedConsensusFeatures
   
   # Create advanced consensus features
   advanced_features = AdvancedConsensusFeatures(consensus_instance)
   
   # Select optimal validators
   validators = await advanced_features.select_optimal_validators(
       num_validators=10,
       exclude_nodes={"current_node"}
   )
   ```

#### Configuration

```python
# Default configuration
config = {
    "selection_window": 50,        # Recent selection history window
    "underrepresentation_threshold": 0.2,  # 20% threshold for boosting
    "score_weights": {
        "response_time": 0.3,
        "success_rate": 0.3,
        "reliability": 0.2,
        "recency": 0.2
    }
}
```

### Adaptive Timeouts

The framework automatically adjusts consensus timeouts based on network conditions and historical performance.

#### Features

1. **Phase-Specific Timeouts**
   - PROPOSING: Base timeout (default 5s)
   - PREPARING: Base timeout (default 5s)
   - COMMITTING: Base timeout (default 5s)
   - FINALIZING: Half base timeout (default 2.5s)

2. **Adaptive Adjustment**
   - Exponential smoothing for stable adjustments
   - Configurable adjustment factor (default 0.1)
   - Bounds enforcement (1s minimum, 30s maximum)

3. **Usage Example**
   ```python
   # Start phase timer
   advanced_features.start_consensus_phase_timer("PREPARING")
   
   # ... perform consensus operations ...
   
   # Stop timer and adjust timeout
   advanced_features.stop_consensus_phase_timer("PREPARING", success=True)
   
   # Get current adaptive timeout
   timeout = advanced_features.get_adaptive_timeout("PREPARING")
   ```

#### Configuration

```python
# Timeout configuration
timeout_config = {
    "base_timeout": 5.0,      # Base timeout in seconds
    "min_timeout": 1.0,       # Minimum timeout
    "max_timeout": 30.0,      # Maximum timeout
    "adjustment_factor": 0.1, # How quickly to adjust
    "smoothing_factor": 0.3   # Exponential smoothing
}
```

---

## Error Handling and Recovery

### Recovery Strategies

The framework implements five recovery strategies for different error scenarios:

1. **RETRY**: Simple retry with exponential backoff
2. **FALLBACK**: Use alternative implementation
3. **RESTART**: Restart the component
4. **DEGRADED**: Operate with reduced functionality
5. **FAILSAFE**: Enter safe minimal operation mode

### Circuit Breaker Pattern

Prevents cascading failures by temporarily stopping requests to failing components.

```python
from aegis.core.error_handling import CircuitBreaker

# Create circuit breaker
breaker = CircuitBreaker(
    failure_threshold=5,  # Trip after 5 failures
    timeout=60.0,         # 60s timeout before retry
    expected_exception=Exception
)

# Use in component
if breaker.allow_request():
    try:
        # Perform operation
        result = risky_operation()
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise
```

### Error Recovery Manager

Centralized error handling and recovery system.

```python
from aegis.core.error_handling import ErrorRecoveryManager, ErrorSeverity

# Create error manager
error_manager = ErrorRecoveryManager()

# Register component
error_manager.register_component("blockchain")

# Handle error
try:
    blockchain_operation()
except Exception as e:
    await error_manager.handle_error(
        component="blockchain",
        error=e,
        severity=ErrorSeverity.MEDIUM
    )
```

---

## Consensus Visualization

### Real-time Monitoring

The framework provides real-time visualization of consensus performance metrics.

#### Features

1. **Performance Metrics**
   - Transaction throughput (TPS)
   - Network latency (ms)
   - Success rate (0-1)
   - Active validator count

2. **Validator Performance**
   - Response time tracking
   - Success rate monitoring
   - Reputation scoring
   - Participation rates

3. **Anomaly Detection**
   - Throughput drops (>30% decrease)
   - Latency spikes (2x increase)
   - Success rate drops (>10% decrease)

### Usage

```python
from aegis.monitoring.consensus_visualizer import ConsensusVisualizer

# Create visualizer
visualizer = ConsensusVisualizer(update_interval=1.0)

# Update with metrics
metrics = ConsensusMetrics(
    timestamp=time.time(),
    view_number=1,
    sequence_number=100,
    active_validators=20,
    total_nodes=25,
    consensus_state="PREPARING",
    throughput=500.0,
    latency=45.2,
    success_rate=0.98
)

visualizer.update_metrics(metrics)

# Start visualization (requires matplotlib)
visualizer.start_visualization()
```

### Web Dashboard

Interactive web-based dashboard using Plotly.

```python
from aegis.monitoring.consensus_visualizer import WebConsensusDashboard

# Create web dashboard
dashboard = WebConsensusDashboard()

# Generate interactive dashboard
fig = dashboard.create_dashboard(visualizer)
fig.show()
```

---

## Performance Optimization

### Memory Management

1. **Circular Buffers**
   - Fixed-size deques for metric history
   - Automatic eviction of old data
   - Configurable buffer sizes

2. **Efficient Data Structures**
   - defaultdict for sparse data
   - deque for FIFO operations
   - Sets for fast membership testing

### Computational Efficiency

1. **Algorithmic Optimizations**
   - O(n log n) sorting for validator selection
   - Exponential smoothing for timeout adjustments
   - Incremental metric updates

2. **Concurrency**
   - Async/await for non-blocking operations
   - Background tasks for maintenance
   - Lock-free data structures where possible

### Scalability Features

1. **Horizontal Scaling**
   - Dynamic validator selection scales with network size
   - Load distribution across validators
   - Network-aware node discovery

2. **Resource Management**
   - Adaptive timeout scaling
   - Performance-based resource allocation
   - Automatic degradation under load

---

## Security Integration

### Rate Limiting

Built-in rate limiting to prevent abuse and DDoS attacks.

```python
from aegis.security.middleware import SecurityMiddleware

# Create security middleware
security = SecurityMiddleware()

# Check rate limit
allowed, message = security.check_request_security(
    client_id="192.168.1.100",
    endpoint="/consensus/propose",
    params={"data": "proposal_data"}
)

if not allowed:
    raise Exception(f"Rate limited: {message}")
```

### Input Validation

Comprehensive input validation and sanitization.

```python
from aegis.security.middleware import InputValidator

# Create validator
validator = InputValidator()

# Validate string input
is_valid, error = validator.validate_string(
    value="user_input_123",
    field_type="alphanumeric",
    min_length=3,
    max_length=50
)

# Sanitize input
safe_string = validator.sanitize_string("<script>alert('xss')</script>Hello")
```

### Threat Detection

Automatic detection of suspicious activity patterns.

```python
# Detect suspicious activity
is_suspicious = security.detect_suspicious_activity(
    client_id="suspicious_client",
    threshold=10,
    window_seconds=60
)

if is_suspicious:
    security.block_client("suspicious_client", duration_seconds=3600)
```

---

## API Reference

### AdvancedConsensusFeatures

#### Methods

- `select_optimal_validators(num_validators, exclude_nodes)` - Select best validators
- `start_consensus_phase_timer(phase)` - Start phase timing
- `stop_consensus_phase_timer(phase, success)` - Stop phase timing
- `get_adaptive_timeout(phase)` - Get current timeout
- `record_validation_result(node_id, success, response_time, message_type)` - Record validation
- `record_message_latency(message_type, latency)` - Record message latency
- `get_performance_report()` - Get performance statistics

### ErrorRecoveryManager

#### Methods

- `handle_error(component, error, severity, context)` - Handle and recover from error
- `register_component(component_name)` - Register component for monitoring
- `register_recovery_strategy(error_pattern, strategy)` - Register recovery strategy
- `get_error_statistics()` - Get error statistics
- `clear_error_history(component)` - Clear error history

### ConsensusVisualizer

#### Methods

- `update_metrics(metrics)` - Update with new metrics
- `update_validator_metrics(validator_metrics)` - Update validator metrics
- `add_alert(alert_type, message, severity)` - Add alert
- `get_performance_summary()` - Get performance summary
- `get_validator_performance()` - Get validator rankings
- `detect_anomalies()` - Detect performance anomalies
- `generate_report()` - Generate comprehensive report

---

## Best Practices

### Consensus Optimization

1. **Validator Selection**
   - Regularly update performance metrics
   - Use appropriate selection window
   - Monitor validator health

2. **Timeout Management**
   - Start/stop timers for each phase
   - Monitor timeout adjustments
   - Tune adjustment factors for stability

### Error Handling

1. **Recovery Strategy Selection**
   - Match strategy to error type
   - Consider component criticality
   - Test recovery procedures

2. **Monitoring**
   - Track error patterns
   - Monitor recovery success
   - Alert on critical failures

### Performance Monitoring

1. **Metrics Collection**
   - Collect relevant performance data
   - Update metrics regularly
   - Store historical data

2. **Visualization**
   - Use real-time dashboards
   - Set up anomaly detection
   - Configure alerts for issues

### Security

1. **Rate Limiting**
   - Configure appropriate limits
   - Monitor abuse patterns
   - Adjust limits based on usage

2. **Input Validation**
   - Validate all external inputs
   - Sanitize user data
   - Use allowlists where possible

---

## Troubleshooting

### Common Issues

1. **Validator Selection Returns Empty List**
   - Check network connectivity
   - Verify node registrations
   - Review security filtering

2. **Timeouts Too Short/Long**
   - Monitor network latency
   - Adjust base timeout values
   - Check timeout adjustment logs

3. **High Error Rates**
   - Review error statistics
   - Check component health
   - Verify recovery strategies

### Performance Tuning

1. **Throughput Optimization**
   - Increase validator count
   - Optimize network configuration
   - Tune timeout settings

2. **Latency Reduction**
   - Improve network connectivity
   - Optimize data serialization
   - Reduce validation overhead

3. **Resource Usage**
   - Monitor memory consumption
   - Adjust buffer sizes
   - Optimize data structures

---

## Changelog

### Version 2.1.0 (2025-10-24)
- Initial release of advanced consensus features
- Dynamic validator selection implementation
- Adaptive timeout management
- Comprehensive error handling system
- Real-time consensus visualization
- Security integration enhancements
