# ⚛️ IACE v2.0 - Fully Python-Native QECS Orchestration Engine

## Overview

IACE v2.0 is a fully Python-native orchestration engine for the Quantum Economic Coherence System (QECS) v1.5. It replaces all Bash dependencies with internal Python orchestration and provides real-time KPI streaming, live anomaly detection, self-healing capabilities, and dashboard telemetry integration.

## Key Features

- **Phase-Level Deterministic Exit Codes**: Enables CI/CD and AGI orchestration integration
- **Real-Time KPI Streaming & Logging**: All metrics available internally for dashboard or AGI consumption
- **Live Anomaly Detection and Isolation**: Monitors system health and triggers alerts
- **Self-Healing Capabilities**: Automatic tuning of HARU λ, CAF α, and I_eff parameters
- **Dashboard Telemetry Integration**: Real-time visualization of field coherence and policy adjustments
- **AGI Improvement Proposal Generation**: Automated system optimization reports
- **Autonomous Gravity Well Daemon**: Continuous monitoring and node isolation

## Architecture

```
IACE v2.0 Orchestration Engine
├── Core Orchestration (iace_v2_orchestrator.py)
│   ├── Phase I: Core System Initialization
│   ├── Phase II/III: Transaction & Field Security
│   ├── Gravity Well Monitoring Daemon
│   └── Phase IV: AGI Reporting & Self-Healing
├── Telemetry System (src/monitoring/telemetry_streamer.py)
│   ├── Real-time KPI Streaming
│   ├── Historical Trend Storage
│   └── Anomaly Detection
├── Dashboard Integration (src/api/routes/telemetry.py)
│   ├── Current Metrics API
│   ├── Historical Data API
│   └── System Status API
└── Visualization (ui-dashboard/iace_dashboard.html)
    ├── Real-time Metrics Display
    ├── Historical Trend Charts
    └── Orchestration Phase Tracking
```

## Components

### 1. Core Orchestration Engine (`iace_v2_orchestrator.py`)

The main orchestration engine that executes the four phases of QECS validation:

- **Phase I**: Initializes core modules and verifies system streaming
- **Phase II/III**: Runs ledger integrity, entropy burn, and field security tests
- **Gravity Well Daemon**: Continuously monitors cluster coherence and isolates nodes when necessary
- **Phase IV**: Generates AGI improvement proposals and applies self-healing corrections

### 2. Telemetry Streamer (`src/monitoring/telemetry_streamer.py`)

Provides real-time telemetry streaming and historical KPI trend tracking:

- Subscribes to system metrics updates
- Maintains historical data in JSON format
- Provides anomaly detection capabilities
- Supports real-time dashboard integration

### 3. Telemetry API (`src/api/routes/telemetry.py`)

Flask blueprint that exposes telemetry data to dashboards:

- `/api/telemetry/current` - Current telemetry data
- `/api/telemetry/history/<metric_name>` - Historical data for specific metrics
- `/api/telemetry/metrics` - List of available metrics
- `/api/telemetry/status` - Telemetry system status

### 4. Dashboard (`ui-dashboard/iace_dashboard.html`)

Interactive web dashboard for visualizing QECS metrics:

- Real-time KPI displays
- Historical trend charts
- Orchestration phase tracking
- System status monitoring

## Installation

1. Ensure all QECS components are properly installed:
   ```bash
   pip install -r requirements.txt
   ```

2. No additional Bash dependencies are required - fully Python-native

## Usage

### Running the Orchestration Engine

```bash
python run_iace_v2.py
```

### Starting the Dashboard Server

```bash
python src/api/main_api.py
```

Then open `ui-dashboard/iace_dashboard.html` in a web browser.

## API Endpoints

### Telemetry Endpoints

- `GET /api/telemetry/current` - Get current telemetry data
- `GET /api/telemetry/history/<metric_name>` - Get historical data for a metric
- `GET /api/telemetry/metrics` - Get list of available metrics
- `GET /api/telemetry/status` - Get telemetry system status

### Existing QECS Endpoints

- `GET /field/curvature_metrics` - Current curvature metrics
- `GET /system/status` - System status
- `POST /ledger/commit` - Commit transactions

## Integration with AGI Systems

IACE v2.0 is designed for seamless integration with AGI governance systems:

1. **Deterministic Exit Codes**: Enable automated decision-making
2. **JSON Reporting**: Machine-readable improvement proposals
3. **Real-time Telemetry**: Continuous system state monitoring
4. **Self-Healing**: Autonomous system optimization

## Files Overview

- `iace_v2_orchestrator.py` - Main orchestration engine
- `run_iace_v2.py` - Runner script
- `src/monitoring/telemetry_streamer.py` - Telemetry streaming system
- `src/api/routes/telemetry.py` - Telemetry API endpoints
- `ui-dashboard/iace_dashboard.html` - Interactive dashboard
- `src/api/main_api.py` - Updated main API with telemetry integration

## Benefits Over v1.5

1. **No Bash Dependencies**: Fully Python-executable, platform-agnostic
2. **Live KPI Streaming**: All metrics available internally for dashboard or AGI consumption
3. **Self-Healing**: Automatic parameter corrections on KPI deviations
4. **Autonomous Gravity Well Daemon**: Real-time monitoring without external scripts
5. **AGI-Ready Reporting**: JSON reports with metrics, anomalies, and optimization vectors
6. **Deterministic Exit Codes**: Enables CI/CD and AGI orchestration integration
7. **Closed Governance Loop**: Predictive, proactive monetary and field security governance

IACE v2.0 represents a significant advancement in autonomous quantum economic system orchestration.