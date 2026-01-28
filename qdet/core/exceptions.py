class QDETError(Exception):
    """Base exception class for QDET library."""
    pass


class QuantumCapacityError(QDETError):
    """Raised when data exceeds qubit limits."""
    pass


class DriftDetectedError(QDETError):
    """Raised by governance checks when data drift is found."""
    pass
