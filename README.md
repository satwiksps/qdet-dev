

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen)](https://github.com/meow/quantum-data-engineering)

## `QDET`: Quantum Data Engineering Toolkit

**QDET** is a professional, production-ready Python library designed to seamlessly integrate quantum computing with classical data engineering and machine learning workflows. QDET bridges the gap between traditional data science and quantum machine learning, enabling researchers and practitioners to build hybrid quantum-classical applications without requiring deep expertise in quantum computing.

## What is QDET?

QDET is an enterprise-grade toolkit that provides:

- **Quantum-Classical Hybrid Workflows**: Seamlessly combine quantum algorithms with classical preprocessing and post-processing
- **Modular Architecture**: 7-layer design with clear separation of concerns
- **Production-Ready Code**: 665+ test cases, full type hints, comprehensive documentation
- **Enterprise Features**: Drift monitoring, privacy preservation, cost tracking, audit logging
- **Easy Integration**: Works with Qiskit, scikit-learn, Pandas, NumPy, and other popular libraries



## Features

### 1. Data Processing & Ingestion

**Comprehensive Data Loading**
- Multi-format data support (CSV, Parquet, SQL databases)
- Streaming data buffers for real-time processing
- Automatic data validation and type checking
- Memory-efficient processing for large datasets

```python
from qdet.connectors import QuantumDataLoader, QuantumSQLLoader

# Load from multiple formats
loader = QuantumDataLoader()
data_csv = loader.load_csv("data.csv")
data_parquet = loader.load_parquet("data.parquet")

# Load from database
sql_loader = QuantumSQLLoader(connection_string="...")
data_sql = sql_loader.query("SELECT * FROM quantum_data")
```

### 2. Feature Engineering & Preprocessing

**Advanced Transformation Capabilities**
- Dimensionality reduction (PCA, projections)
- Feature scaling (Standard, Min-Max, Quantum normalization)
- Feature selection and extraction
- Outlier detection and removal
- Data balancing and resampling
- Categorical encoding (One-hot, Target, Frequency, Binning)

```python
from qdet.transforms import (
    QuantumPCA, FeatureScaler, FeatureSelector,
    OutlierRemover, DataBalancer, CategoricalEncoder
)

# Complete preprocessing pipeline
pca = QuantumPCA(n_components=10)
X_reduced = pca.fit_transform(X)

scaler = FeatureScaler(method="quantum_aware")
X_scaled = scaler.fit_transform(X_reduced)

selector = FeatureSelector(method="importance", n_features=8)
X_selected = selector.fit_transform(X_scaled, y)

balancer = DataBalancer(strategy="smote")
X_balanced, y_balanced = balancer.fit_resample(X_selected, y)
```

### 3. Quantum Encoding

**Multiple Encoding Strategies**
- **Amplitude Encoding**: Encode data as quantum state amplitudes
- **Angle Encoding**: Map features to rotation angles
- **IQP Encoding**: Instantaneous Quantum Polynomial circuits
- **Composite Encoding**: Combine multiple encoding schemes
- **Adaptive Encoding**: Learn optimal encoding parameters

```python
from qdet.encoders import (
    AmplitudeEncoder, AngleEncoder,
    IQPEncoder, CompositeEncoder
)

# Amplitude encoding
amplitude_enc = AmplitudeEncoder(n_qubits=4)
quantum_state = amplitude_enc.encode(classical_features)

# Multi-scheme encoding
composite_enc = CompositeEncoder(
    encoders=[AmplitudeEncoder(2), AngleEncoder(2)],
    weights=[0.6, 0.4]
)
hybrid_state = composite_enc.encode(features)
```

### 4. Quantum Machine Learning Algorithms

**Comprehensive Algorithm Suite**
- Classification (SVM, Neural Networks, Ensembles)
- Regression (Linear, Kernel methods)
- Clustering (K-Means, Spectral)
- Anomaly Detection
- Time Series Analysis
- Variational Algorithms (VQE, QAOA)
- Autoencoding

```python
from qdet.analytics import (
    QuantumSVC, QuantumRegressor, QuantumCluster,
    QuantumAnomalyDetector, QuantumAutoencoder
)

# Quantum classification
classifier = QuantumSVC(kernel="quantum", backend="aer_simulator")
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)

# Quantum clustering
clusterer = QuantumCluster(n_clusters=3, algorithm="kmeans_q")
cluster_labels = clusterer.fit_predict(X)

# Anomaly detection
anomaly_detector = QuantumAnomalyDetector(
    contamination=0.1, method="isolation_forest_q"
)
anomaly_scores = anomaly_detector.fit_predict(X)
```

### 5. Distributed Computing & Resource Management

**Efficient Quantum Circuit Management**
- Circuit compilation and optimization
- Multi-backend support (Qiskit Aer, IBM Runtime, etc.)
- Error mitigation strategies
- Resource scheduling and allocation
- Cost estimation before execution

```python
from qdet.compute import (
    CircuitCompiler, BackendManager,
    ErrorMitigation, ResourceScheduler, CostEstimator
)

# Manage quantum backends
backend_mgr = BackendManager()
backend_mgr.add_backend("aer", "qasm_simulator")
backend_mgr.add_backend("ibm", "ibm_qx5")

# Compile circuits for specific backend
compiler = CircuitCompiler()
optimized_circuit = compiler.compile(
    circuit, backend="aer", optimization_level=3
)

# Estimate execution cost
estimator = CostEstimator()
cost = estimator.estimate(circuit, backend="ibm")
print(f"Estimated cost: ${cost:.2f}")

# Apply error mitigation
mitigator = ErrorMitigation(method="zne")
result = mitigator.execute(circuit, backend)
```

### 6. Governance & Monitoring

**Enterprise-Grade Monitoring**
- **Drift Detection**: Monitor data and model drift
- **Privacy Protection**: Implement differential privacy, data anonymization
- **Audit Logging**: Track all operations for compliance
- **Cost Monitoring**: Track and optimize quantum resource spending
- **Performance Validation**: Continuous model monitoring
- **Integrity Checks**: Data quality validation

```python
from qdet.governance import (
    DriftDetector, PrivacyPreserver, AuditLogger,
    CostMonitor, ModelValidator, DataIntegrityChecker
)

# Monitor data drift
drift_detector = DriftDetector(method="ks_test", threshold=0.05)
has_drift, p_value = drift_detector.check(X_current, X_reference)

# Apply privacy techniques
privacy_preserver = PrivacyPreserver(method="differential_privacy")
X_private = privacy_preserver.fit_transform(X, epsilon=1.0)

# Track operations
audit_logger = AuditLogger(log_file="audit.log")
audit_logger.log_model_training("SVC", {"C": 1.0, "kernel": "rbf"})
audit_logger.log_prediction(model_id, features, prediction)

# Monitor costs
cost_monitor = CostMonitor()
cost_monitor.record_execution(circuit, backend="ibm", time=1.2)
total_cost = cost_monitor.get_total_cost()

# Validate data
validator = DataIntegrityChecker()
is_valid = validator.validate(X, y)
report = validator.get_validation_report()
```

## Quick Start

### Example 1: Basic Classification

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qdet.transforms import FeatureScaler
from qdet.encoders import AmplitudeEncoder
from qdet.analytics import QuantumSVC

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess: Scale features
scaler = FeatureScaler(method="standard")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quantum classification
classifier = QuantumSVC(
    kernel="quantum",
    backend="aer_simulator",
    n_qubits=4,
    shots=1000
)
classifier.fit(X_train_scaled, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test_scaled)
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
```

### Example 2: Anomaly Detection

```python
import numpy as np
from qdet.transforms import QuantumPCA, FeatureScaler
from qdet.governance import DriftDetector
from qdet.analytics import QuantumAnomalyDetector

# Generate synthetic data
X_normal = np.random.randn(100, 20)
X_test = np.vstack([
    X_normal,
    np.random.uniform(5, 10, (5, 20))  # Anomalies
])

# Preprocess
scaler = FeatureScaler(method="quantum_aware")
X_scaled = scaler.fit_transform(X_test)

# Reduce dimensions
pca = QuantumPCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled)

# Detect anomalies
anomaly_detector = QuantumAnomalyDetector(
    contamination=0.05,
    method="isolation_forest_q"
)
anomalies = anomaly_detector.fit_predict(X_reduced)
print(f"Detected {anomalies.sum()} anomalies")

# Check for data drift
drift_detector = DriftDetector(method="ks_test")
has_drift, p_value = drift_detector.check(X_scaled[:50], X_scaled[50:])
print(f"Data drift detected: {has_drift} (p-value: {p_value:.4f})")
```

### Example 3: Time Series Forecasting

```python
import numpy as np
from qdet.analytics import QuantumTimeSeries
from qdet.transforms import QuantumPCA
from qdet.governance import DriftDetector

# Generate time series data
np.random.seed(42)
time_series = np.cumsum(np.random.randn(200, 1))

# Create sequences
X = np.array([time_series[i:i+10] for i in range(len(time_series)-10)])
y = time_series[10:, 0]

# Reduce dimensionality
pca = QuantumPCA(n_components=5)
X_reduced = pca.fit_transform(X)

# Quantum time series model
ts_model = QuantumTimeSeries(
    window_size=10,
    backend="aer_simulator",
    n_qubits=5
)
ts_model.fit(X_reduced, y)

# Forecast
forecast = ts_model.predict(X_reduced[-1:], steps=5)
print(f"5-step forecast: {forecast}")

# Monitor drift in time series
drift_detector = DriftDetector()
has_drift, _ = drift_detector.check(X_reduced[:100], X_reduced[100:])
print(f"Time series drift: {has_drift}")
```

## Architecture

### 7-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer                                      │
│  (User code, experiments, ML workflows)                 │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  ANALYTICS LAYER                                        │
│  • Classification (SVC, NN, Ensemble)                   │
│  • Regression (Linear, Kernel, Ridge)                   │
│  • Clustering (K-Means, Spectral)                       │
│  • Anomaly Detection (Isolation, LOF)                   │ 
│  • Time Series (LSTM, ARIMA)                            │
│  • Variational Algorithms (VQE, QAOA)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  GOVERNANCE LAYER                                       │
│  • Drift Detection & Monitoring                         │
│  • Privacy Preservation (Differential Privacy)          │
│  • Audit Logging & Compliance                           │
│  • Cost Tracking & Optimization                         │
│  • Model Validation & Monitoring                        │ 
│  • Data Integrity Checks                                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  COMPUTE LAYER                                          │
│  • Circuit Compilation & Optimization                   │
│  • Multi-Backend Support                                │
│  • Error Mitigation Strategies                          │
│  • Resource Allocation & Scheduling                     │
│  • Cost Estimation                                      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  ENCODERS LAYER                                         │
│  • Amplitude Encoding                                   │
│  • Angle/Phase Encoding                                 │
│  • IQP Encoding                                         │
│  • Composite Encoding                                   │
│  • Adaptive Encoding                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  TRANSFORMS LAYER                                       │
│  • Feature Scaling (Standard, Min-Max, Robust)          │
│  • Feature Selection (Importance, Correlation)          │
│  • Dimensionality Reduction (PCA, Projections)          │
│  • Categorical Encoding (One-hot, Target, Freq)         │
│  • Outlier Removal & Data Balancing                     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  CONNECTORS LAYER                                       │
│  • Data Loading (CSV, Parquet, SQL)                     │
│  • Data Validation & Type Checking                      │
│  • Streaming Data Buffers                               │
│  • Caching & Serialization                              │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  CORE LAYER                                             │
│  • Base Classes & Exceptions                            │
│  • Abstract Interfaces                                  │
│  • Common Utilities                                     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Through Layers

```
Raw Data
   ↓
[CONNECTORS] Load & Validate
   ↓
[TRANSFORMS] Preprocess & Feature Engineer
   ↓
[ENCODERS] Convert to Quantum Representations
   ↓
[COMPUTE] Optimize & Execute Circuits
   ↓
[ANALYTICS] Run Quantum ML Algorithms
   ↓
[GOVERNANCE] Monitor, Track, Audit
   ↓
Results & Insights
```

## Module Documentation

### 1. Core Layer (`qdet.core`)

**Base Classes**
```python
from qdet.core import BaseReducer, BaseEncoder, BaseQuantumEstimator

class CustomReducer(BaseReducer):
    """Custom dimensionality reduction."""
    def fit(self, X, y=None):
        # Implementation
        return self
    
    def transform(self, X):
        # Implementation
        return X_transformed
```

### 2. Connectors Layer (`qdet.connectors`)

**Data Loading**
```python
from qdet.connectors import QuantumDataLoader, QuantumSQLLoader

# CSV loading
loader = QuantumDataLoader()
df = loader.load_csv("data.csv", dtype_mapping={'price': float})

# SQL loading
sql_loader = QuantumSQLLoader(
    db_type="postgresql",
    host="localhost",
    port=5432,
    database="quantum_db"
)
data = sql_loader.query("SELECT * FROM quantum_features WHERE quality > 0.8")

# Parquet loading (efficient)
data_parquet = loader.load_parquet("large_dataset.parquet")
```

**Data Validation**
```python
from qdet.connectors import DataValidator

validator = DataValidator()
issues = validator.validate(X, y)
print(f"Found {len(issues)} data issues")
for issue in issues:
    print(f"  - {issue}")
```

### 3. Transforms Layer (`qdet.transforms`)

**Complete Preprocessing Pipeline**
```python
from qdet.transforms import (
    QuantumPCA, FeatureScaler, FeatureSelector,
    OutlierRemover, DataBalancer, CategoricalEncoder,
    QuantumNormalizer, RangeNormalizer
)
from sklearn.pipeline import Pipeline

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', FeatureScaler(method='standard')),
    ('pca', QuantumPCA(n_components=10)),
    ('selector', FeatureSelector(n_features=8)),
    ('outlier_remover', OutlierRemover(contamination=0.05)),
    ('balancer', DataBalancer(strategy='smote')),
    ('normalizer', QuantumNormalizer())
])

X_processed = preprocessing_pipeline.fit_transform(X, y)
```

**Specific Transformers**
```python
# Feature Scaling
scaler = FeatureScaler(method='quantum_aware', feature_range=(0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction
pca = QuantumPCA(n_components=5, n_iter=100)
X_reduced = pca.fit_transform(X)

# Categorical Encoding
encoder = CategoricalEncoder(method='target', target_type='binary')
X_encoded = encoder.fit_transform(X_categorical, y)

# Outlier Detection and Removal
outlier_detector = OutlierRemover(method='isolation_forest', contamination=0.1)
X_clean = outlier_detector.fit_transform(X)

# Data Balancing
balancer = DataBalancer(strategy='adasyn', sampling_strategy=0.5)
X_balanced, y_balanced = balancer.fit_resample(X, y)
```

### 4. Encoders Layer (`qdet.encoders`)

**Available Encoding Methods**
```python
from qdet.encoders import (
    AmplitudeEncoder, AngleEncoder, IQPEncoder,
    CompositeEncoder, StatevectorEncoder
)

# Amplitude Encoding: Encode data as quantum state
amplitude_enc = AmplitudeEncoder(n_qubits=4)
state = amplitude_enc.encode([0.5, 0.3, 0.2])

# Angle Encoding: Map features to rotation angles
angle_enc = AngleEncoder(n_qubits=3, encoding_style="Z-rotation")
state = angle_enc.encode([0.1, 0.2, 0.3])

# IQP Encoding: Instantaneous Quantum Polynomial
iqp_enc = IQPEncoder(n_qubits=4, reps=2)
state = iqp_enc.encode(features)

# Composite Encoding: Combine multiple schemes
composite = CompositeEncoder(
    encoders=[
        AmplitudeEncoder(n_qubits=2),
        AngleEncoder(n_qubits=2)
    ],
    backend='aer_simulator'
)
state = composite.encode(features)
```

### 5. Analytics Layer (`qdet.analytics`)

**Quantum Machine Learning Models**
```python
from qdet.analytics import (
    QuantumSVC, QuantumRegressor, QuantumCluster,
    QuantumAnomalyDetector, QuantumAutoencoder,
    QuantumNeuralNetwork, QuantumEnsemble
)

# Classification
classifier = QuantumSVC(
    kernel='quantum',
    C=1.0,
    backend='aer_simulator',
    shots=1000,
    random_state=42
)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Regression
regressor = QuantumRegressor(
    algorithm='vqr',  # Variational Quantum Regression
    n_qubits=4,
    depth=3,
    learning_rate=0.01
)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Clustering
clusterer = QuantumCluster(
    n_clusters=3,
    algorithm='kmeans_q',
    n_qubits=3,
    max_iter=100
)
labels = clusterer.fit_predict(X)

# Anomaly Detection
anomaly_detector = QuantumAnomalyDetector(
    method='qnn_autoencoder',
    contamination=0.05,
    threshold_percentile=95
)
anomaly_scores = anomaly_detector.fit_predict(X)

# Neural Network
nn = QuantumNeuralNetwork(
    n_qubits=4,
    n_layers=3,
    optimizer='adam',
    learning_rate=0.01
)
nn.fit(X_train, y_train, epochs=10, batch_size=32)

# Ensemble Methods
ensemble = QuantumEnsemble(
    estimators=[
        QuantumSVC(kernel='quantum'),
        QuantumSVC(kernel='rbf'),
        QuantumRegressor()
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
```

### 6. Compute Layer (`qdet.compute`)

**Backend Management**
```python
from qdet.compute import BackendManager, CircuitCompiler, CostEstimator

# Manage quantum backends
backend_mgr = BackendManager()
backend_mgr.add_backend('aer', 'qasm_simulator')
backend_mgr.add_backend('aer_statevector', 'statevector_simulator')
backend_mgr.add_backend('ibm_runtime', 'ibm_qx5')

# Get available backends
available = backend_mgr.list_backends()
print(f"Available backends: {available}")

# Compile circuits
compiler = CircuitCompiler()
optimized = compiler.compile(
    circuit,
    backend='aer',
    optimization_level=3
)

# Estimate costs
estimator = CostEstimator()
cost = estimator.estimate(circuit, backend='ibm_runtime')
print(f"Estimated cost: ${cost:.2f}")
```

**Error Mitigation**
```python
from qdet.compute import ErrorMitigation

# Zero Noise Extrapolation (ZNE)
mitigator = ErrorMitigation(method='zne', num_factors=5)
result = mitigator.execute(circuit, backend)

# Measurement Error Mitigation
mitigator_measure = ErrorMitigation(method='measurement_error')
result = mitigator_measure.execute(circuit, backend)

# Probabilistic Error Cancellation (PEC)
mitigator_pec = ErrorMitigation(method='pec', num_samples=1000)
result = mitigator_pec.execute(circuit, backend)
```

### 7. Governance Layer (`qdet.governance`)

**Drift Detection**
```python
from qdet.governance import DriftDetector

detector = DriftDetector(method='ks_test', threshold=0.05)
has_drift, p_value = detector.check(X_reference, X_current)

if has_drift:
    print(f"Data drift detected! p-value: {p_value:.4f}")
    print("Recommendation: Retrain model")
```

**Privacy Protection**
```python
from qdet.governance import PrivacyPreserver

# Differential privacy
privacy = PrivacyPreserver(method='differential_privacy')
X_private = privacy.fit_transform(X, epsilon=1.0, delta=0.01)

# Data anonymization
privacy_anon = PrivacyPreserver(method='anonymization')
X_anonymous = privacy_anon.fit_transform(X)

# Federated learning support
privacy_federated = PrivacyPreserver(method='federated')
X_federated = privacy_federated.fit_transform(X)
```

**Audit Logging**
```python
from qdet.governance import AuditLogger

logger = AuditLogger(log_file='audit.log')
logger.log_model_training(
    model_name='QuantumSVC',
    parameters={'C': 1.0, 'kernel': 'quantum'},
    timestamp=True
)
logger.log_prediction(model_id, features, prediction, confidence)

# Get audit report
report = logger.generate_report()
print(report)
```

**Cost Monitoring**
```python
from qdet.governance import CostMonitor

monitor = CostMonitor()
monitor.record_execution(circuit, backend='ibm', shots=1000, time=1.2)
monitor.record_api_call(service='ibm_quantum', cost=5.00)

total_cost = monitor.get_total_cost()
cost_breakdown = monitor.get_cost_breakdown()
print(f"Total cost: ${total_cost:.2f}")
print(f"Cost by service:\n{cost_breakdown}")
```

## Other Usage

### Building Custom Quantum Circuits

```python
from qdet.compute import QuantumCircuitBuilder
import numpy as np

builder = QuantumCircuitBuilder(n_qubits=4)

# Build custom circuit
circuit = builder.h_layer(range(4))  # Hadamard on all qubits
circuit = builder.entangle([(0, 1), (2, 3)])  # Entangle pairs
circuit = builder.rotation_layer(
    angles=np.random.randn(4),
    axis='Z'
)

# Execute and get results
results = builder.execute(circuit, backend='aer_simulator', shots=1000)
print(f"Measurement results:\n{results}")
```

### Hyperparameter Optimization

```python
from qdet.analytics import QuantumSVC
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['quantum', 'rbf', 'poly'],
    'depth': [1, 2, 3]
}

# Grid search
model = QuantumSVC()
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
```

### Multi-Backend Evaluation

```python
from qdet.analytics import QuantumSVC
from qdet.compute import BackendManager

backends = ['aer_simulator', 'qasm_simulator', 'statevector_simulator']
results = {}

for backend_name in backends:
    model = QuantumSVC(backend=backend_name, shots=1000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    results[backend_name] = accuracy
    
    print(f"{backend_name}: {accuracy:.4f}")

# Compare results
best_backend = max(results, key=results.get)
print(f"\nBest backend: {best_backend} ({results[best_backend]:.4f})")
```

### Real-Time Monitoring

```python
from qdet.governance import DriftDetector, CostMonitor
from qdet.analytics import QuantumSVC
import numpy as np
import time

drift_detector = DriftDetector()
cost_monitor = CostMonitor()

# Training
model = QuantumSVC(backend='aer_simulator')
model.fit(X_train, y_train)

# Real-time prediction and monitoring
for X_batch, y_batch in streaming_data_generator():
    # Make prediction
    start = time.time()
    predictions = model.predict(X_batch)
    exec_time = time.time() - start
    
    # Monitor drift
    has_drift, p_value = drift_detector.check(X_reference, X_batch)
    if has_drift:
        print(f"⚠️ Drift detected (p={p_value:.4f}). Retraining recommended.")
    
    # Track cost
    cost_monitor.record_execution(
        circuit=None,
        backend='aer_simulator',
        shots=1000,
        time=exec_time
    )
    
    # Check cumulative cost
    if cost_monitor.get_total_cost() > 100.0:
        print(f"⚠️ Cost threshold exceeded: ${cost_monitor.get_total_cost():.2f}")
```

## Testing

### Run All Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=qdet --cov-report=html

# Run specific test module
pytest tests/test_analytics/test_classifier.py -v
```
### Writing Tests

```python
import pytest
from qdet.analytics import QuantumSVC

class TestQuantumSVC:
    """Test suite for QuantumSVC."""
    
    def setup_method(self):
        """Setup before each test."""
        self.model = QuantumSVC(backend='aer_simulator')
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        self.model.fit(X, y)
        assert hasattr(self.model, 'classes_')
    
    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        self.model.fit(X[:20], y[:20])
        predictions = self.model.predict(X[20:])
        assert len(predictions) == len(X[20:])
    
    @pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
    def test_parameter_C(self, C, sample_data):
        """Test different C values."""
        X, y = sample_data
        model = QuantumSVC(C=C, backend='aer_simulator')
        model.fit(X, y)
        score = model.score(X, y)
        assert score >= 0.0 and score <= 1.0
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests: `pytest tests/test_your_feature.py`
5. Submit a pull request

