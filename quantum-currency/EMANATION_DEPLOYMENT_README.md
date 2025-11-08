# 🌌 Quantum Currency Emanation Phase Deployment

**Complete deployment and monitoring infrastructure for the Quantum Currency Emanation (Diamond) Phase**

## 📋 Overview

This repository contains all the necessary components to deploy and monitor the Quantum Currency system's Emanation Phase. The implementation includes:

1. **Staging Verification System** - Comprehensive testing before production deployment
2. **Emanation Deployment Monitor** - Continuous monitoring and auto-balancing controller
3. **Containerization** - Docker configuration for deployment
4. **Kubernetes Orchestration** - Production-ready deployment manifests
5. **Metrics Integration** - Prometheus/Grafana compatibility

## 🚀 Quick Start

### Demonstrate the Complete Workflow

```bash
# Run the complete demonstration
python3 demonstrate_emanation_deployment.py
```

This will:
1. Generate sample reports
2. Verify report structure
3. Run staging verification
4. Execute deployment monitoring
5. Show generated files

## 🧪 Staging Verification

### Run Staging Verification

```bash
# Run staging verification with default settings
python3 run_staging_verification.py

# Run with custom parameters
python3 run_staging_verification.py --cycles 5 --interval 30 --output my_report.json
```

### Components Verified

- Harmonic Engine
- Ω-Security Primitives
- Meta-Regulator
- Cosmonic Verification System
- Dashboard Components
- API Endpoints
- Performance Metrics
- Security Audits

## 📊 Emanation Deployment Monitoring

### Run Continuous Monitoring

```bash
# Run monitoring controller (single cycle)
python3 emanation_deploy.py --cycles 1

# Run continuous monitoring
python3 emanation_deploy.py --cycles 0 --interval 60

# Run with custom parameters
python3 emanation_deploy.py --cycles 10 --interval 30 --report-dir /path/to/reports
```

### Key Features

- Real-time metrics monitoring (H_internal, CAF, entropy rate)
- Auto-balance control parameter adjustment
- Alert system integration
- JSON report generation
- Summary reporting

## 🐳 Containerization

### Build Docker Image

```bash
# Build the emanation monitor image
docker build -f Dockerfile.emanation -t quantum-currency/emanation-monitor .
```

### Run Container

```bash
# Run the container
docker run -d --name emanation-monitor \
  -v /path/to/data:/mnt/data \
  quantum-currency/emanation-monitor
```

## ☸️ Kubernetes Deployment

### Deploy to Kubernetes

```bash
# Apply the deployment
kubectl apply -f k8s/emanation-monitor-deployment.yaml

# Apply the cron job
kubectl apply -f k8s/emanation-monitor-cronjob.yaml
```

### Components

- **Deployment**: Continuous monitoring controller
- **CronJob**: Periodic monitoring cycles
- **ServiceAccount**: Secure access permissions
- **RBAC**: Role-based access control

## 📁 Directory Structure

```
quantum-currency/
├── emanation_deploy.py              # Main monitoring controller
├── run_staging_verification.py      # Staging verification runner
├── generate_sample_reports.py       # Sample data generator
├── verify_reports.py                # Report verification tool
├── demonstrate_emanation_deployment.py # Complete workflow demonstration
├── staging_verification.json        # Verification template
├── Dockerfile.emanation             # Container configuration
├── k8s/                             # Kubernetes manifests
│   ├── emanation-monitor-cronjob.yaml
│   └── emanation-monitor-deployment.yaml
├── reports/                         # Generated reports
│   └── staging/                     # Staging verification reports
└── /mnt/data/                       # Monitoring reports (in container)
```

## 🛠️ Key Scripts

### `emanation_deploy.py`
Main monitoring and auto-balance controller:
- Fetches real-time metrics
- Applies auto-balance heuristics
- Generates JSON reports
- Sends alerts when needed

### `run_staging_verification.py`
Comprehensive staging verification system:
- Tests all system components
- Validates performance metrics
- Runs security audits
- Generates detailed reports

### `generate_sample_reports.py`
Creates sample reports for testing:
- Simulates monitoring cycles
- Generates realistic metrics
- Creates summary reports

### `verify_reports.py`
Validates generated reports:
- Checks report structure
- Verifies data integrity
- Ensures completeness

## 📈 Metrics Monitored

### Core Quantum Metrics

- **H_internal**: Internal coherence level (target: ≥ 0.98)
- **CAF**: Coherence Amplification Factor (target: ≥ 1.05)
- **Entropy Rate**: System stability measure (target: ≤ 0.002)
- **Connected Systems**: Network connectivity (target: ≥ 12)
- **Coherence Score**: Overall system coherence (target: ≥ 0.97)

### Control Parameters

- **λ(L)**: Decay modulation parameter
- **m_t**: Modulator computation factor
- **Ω_t**: State vector optimization
- **Ψ**: Coherence metrics refinement

## 🚨 Alerting System

The system includes simulated alerting capabilities:
- Critical threshold alerts
- Warning notifications
- Integration with external systems (Slack/PagerDuty)

## 🔧 Configuration

### Environment Variables

- `QUANTUM_ENV`: Environment (staging/production)
- `PYTHONPATH`: Python module path
- `PROMETHEUS_URL`: Prometheus server URL
- `GRAFANA_URL`: Grafana server URL
- `ALERT_WEBHOOK_URL`: Alert webhook endpoint

### Command Line Options

- `--cycles`: Number of monitoring cycles (0 for continuous)
- `--interval`: Seconds between cycles
- `--report-dir`: Directory for report storage

## 🧪 Testing

### Run All Tests

```bash
# Run staging verification
python3 run_staging_verification.py --cycles 3

# Run monitoring simulation
python3 emanation_deploy.py --cycles 5

# Verify generated reports
python3 verify_reports.py

# Test Prometheus integration
python3 test_prometheus_integration.py
```

## 📊 Report Structure

### Cycle Reports

```json
{
  "cycle": 1,
  "timestamp": "2025-11-08T10:30:00Z",
  "metrics": {
    "h_internal": 0.975,
    "caf": 1.025,
    "entropy_rate": 0.0018,
    "connected_systems": 12,
    "coherence_score": 0.965
  },
  "control_parameters": {
    "lambda_L": 0.55,
    "m_t": 1.02,
    "Omega_t": 0.85,
    "Psi": 0.72
  },
  "adjustments_made": {
    "lambda_L": 0.02,
    "m_t": 0.01,
    "Omega_t": 0.03,
    "Psi": 0.01
  },
  "alerts": [],
  "status": "stable"
}
```

### Summary Reports

```json
{
  "summary": true,
  "timestamp": "2025-11-08T11:00:00Z",
  "total_cycles": 5,
  "average_metrics": {
    "h_internal": 0.978,
    "caf": 1.032,
    "entropy_rate": 0.0017,
    "connected_systems": 11.6,
    "coherence_score": 0.968
  },
  "total_alerts": 0,
  "critical_alerts": 0,
  "final_control_parameters": {
    "lambda_L": 0.58,
    "m_t": 1.05,
    "Omega_t": 0.88,
    "Psi": 0.75
  },
  "status": "stable"
}
```

## 🚀 Production Deployment

### Prerequisites

1. Kubernetes cluster
2. Prometheus monitoring stack
3. Grafana dashboards
4. Alerting system (Slack/PagerDuty)

### Deployment Steps

1. **Configure Secrets**
   ```bash
   kubectl create secret generic alert-webhook --from-literal=url=https://your-alert-webhook
   ```

2. **Deploy Monitoring System**
   ```bash
   kubectl apply -f k8s/emanation-monitor-deployment.yaml
   kubectl apply -f k8s/emanation-monitor-cronjob.yaml
   ```

3. **Verify Deployment**
   ```bash
   kubectl get pods -n quantum-currency
   kubectl logs -n quantum-currency deployment/quantum-currency-emanation-monitor
   ```

## 🆘 Troubleshooting

### Common Issues

1. **Reports not generated**
   - Check write permissions to /mnt/data
   - Verify disk space availability

2. **Metrics not updating**
   - Confirm Prometheus connectivity
   - Check network policies

3. **Alerts not sending**
   - Verify webhook URL configuration
   - Test webhook connectivity

### Logs and Debugging

```bash
# Check deployment logs
kubectl logs -n quantum-currency deployment/quantum-currency-emanation-monitor

# Check cron job logs
kubectl logs -n quantum-currency job/<job-name>

# Port forward for local testing
kubectl port-forward -n quantum-currency svc/quantum-currency-emanation-monitor 8080:8080
```

## 📚 Additional Documentation

- `EMANATION_DEPLOYMENT_SUMMARY.md`: Complete implementation overview
- `EMANATION_PHASE_README.md`: Emanation phase features
- `QUANTUM_CURRENCY_ROADMAP.md`: Project roadmap

---

*Deployment Infrastructure Ready for Emanation Phase* 🌟