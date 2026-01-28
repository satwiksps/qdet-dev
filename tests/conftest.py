"""Pytest Configuration & Shared Fixtures"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def clean_data_2d():
    """Simple clean 2D numpy array. Shape: (100, 2), Range: [0, 1]."""
    np.random.seed(42)
    return np.random.rand(100, 2)


@pytest.fixture
def clean_data_high_dim():
    """High-dimensional clean data for reduction tests. Shape: (200, 50)."""
    np.random.seed(42)
    return np.random.rand(200, 50)


@pytest.fixture
def messy_data_df():
    """DataFrame with NaNs and outliers for imputation tests. Shape: (50, 5)."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.rand(50, 5),
        columns=[f'col_{i}' for i in range(5)]
    )
    df.iloc[0, 0] = np.nan
    df.iloc[5, 2] = np.nan
    df.iloc[10, 4] = np.nan

    df.iloc[15, 1] = 10.0
    df.iloc[20, 3] = -5.0

    return df


@pytest.fixture
def binary_data():
    """Binary classification data. Shape: (100, 4), Labels: Binary (0 or 1)."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification data. Shape: (150, 5), Labels: 3 classes."""
    np.random.seed(42)
    X = np.random.randn(150, 5)
    y = np.random.randint(0, 3, 150)
    return X, y


@pytest.fixture
def clustering_data():
    """Data with natural clusters for clustering tests. Shape: (150, 2)."""
    np.random.seed(42)
    c1 = np.random.randn(50, 2) + [0, 0]
    c2 = np.random.randn(50, 2) + [5, 5]
    c3 = np.random.randn(50, 2) + [10, 0]

    return np.vstack([c1, c2, c3])


@pytest.fixture
def time_series_data():
    """Time series data for drift detection tests. Shape: (100, 3)."""
    np.random.seed(42)
    t = np.arange(100)
    x1 = np.sin(t * 0.1) + np.random.randn(100) * 0.1
    x2 = np.cos(t * 0.1) + np.random.randn(100) * 0.1
    x3 = t / 100.0 + np.random.randn(100) * 0.1
    return np.column_stack([x1, x2, x3])


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.fixture
def simple_circuit():
    """Basic Bell State circuit (2 qubits)."""
    qc = QuantumCircuit(2, name="bell_state")
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.fixture
def parametric_circuit():
    """Parameterized circuit for variational algorithms."""
    qc = QuantumCircuit(2, name="parametric")
    qc.ry(0.5, 0)
    qc.ry(0.5, 1)
    qc.cx(0, 1)
    qc.rz(0.5, 0)
    qc.rz(0.5, 1)
    return qc


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.fixture
def ghz_circuit():
    """GHZ state circuit (3 qubits)."""
    qc = QuantumCircuit(3, name="ghz_state")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def temp_csv_file():
    """Temporary CSV file for I/O tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("feature1,feature2,feature3,label\n")
        for i in range(10):
            f.write(f"{np.random.rand()},{np.random.rand()},{np.random.rand()},{i % 2}\n")
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_parquet_file():
    """Temporary Parquet file for I/O tests."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("PyArrow not installed")

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        temp_path = f.name

    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'label': np.random.randint(0, 2, 100)
    })
    data.to_parquet(temp_path)

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def assert_arrays_close():
    """Helper function to compare numpy arrays with tolerance."""
    def _assert(actual, expected, atol=1e-6):
        np.testing.assert_allclose(actual, expected, atol=atol)
    return _assert


@pytest.fixture
def assert_dataframes_equal():
    """Helper function to compare DataFrames."""
    def _assert(actual, expected):
        pd.testing.assert_frame_equal(actual, expected)
    return _assert


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
