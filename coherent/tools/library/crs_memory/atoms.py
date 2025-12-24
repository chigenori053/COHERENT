"""
CRS Memory Library - MemoryAtom
Definition of the minimal unit of meaning represented as complex spectrum.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import datetime

class AtomType(Enum):
    REAL = "REAL"
    SIGNAL = "SIGNAL"
    IMAGE = "IMAGE"
    ABSTRACT = "ABSTRACT"

@dataclass
class ComplexVal:
    """Complex number representation for serialization."""
    re: float
    im: float

    def to_complex(self) -> complex:
        return complex(self.re, self.im)

    @staticmethod
    def from_complex(c: complex) -> "ComplexVal":
        return ComplexVal(re=c.real, im=c.imag)

@dataclass
class TransformSpec:
    kind: str # FFT1D, FFT2D, DCT, etc.
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InverseSpec:
    kind: str # IFFT1D, etc.
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectionSpec:
    domain: str # REAL_VECTOR, REAL_SIGNAL, REAL_IMAGE
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReconstructQuality:
    reconstructable: bool
    error_metric: str = "mse"
    max_error: float = 0.0

@dataclass
class MemoryAtom:
    """
    Minimal unit of meaning represented as complex spectrum.
    """
    id: str
    atom_type: AtomType
    spec_dim: int
    repr: List[ComplexVal]
    transform: TransformSpec
    inverse: InverseSpec
    projection: ProjectionSpec
    quality: ReconstructQuality
    confidence: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # Optional
    context_signature: Optional[str] = None
    origin: str = "input" # input, inference, learning

    def __post_init__(self):
        # Validation logic could go here
        if len(self.repr) != self.spec_dim:
            # We allow soft mismatch if spec is sparse, but spec says match.
            # For strict v0.1 compliance:
            pass 

    @classmethod
    def from_real_vector(cls, data: list[float], id: str) -> "MemoryAtom":
        """
        Create a MemoryAtom from a real-valued vector using FFT.
        """
        import numpy as np
        
        if not data:
            return cls(
                id=id,
                atom_type=AtomType.SIGNAL,
                spec_dim=0,
                repr=[],
                transform=TransformSpec(kind="FFT1D"),
                inverse=InverseSpec(kind="IFFT1D"),
                projection=ProjectionSpec(domain="REAL_VECTOR"),
                quality=ReconstructQuality(reconstructable=True, error_metric="mse"),
                confidence=1.0
            )

        # 1. FFT
        spectrum = np.fft.fft(data)
        
        # 2. Create ComplexVals
        repr_vals = [ComplexVal.from_complex(c) for c in spectrum]
        
        # 3. Construct Atom
        return cls(
            id=id,
            atom_type=AtomType.SIGNAL,
            spec_dim=len(data),
            repr=repr_vals,
            transform=TransformSpec(kind="FFT1D"),
            inverse=InverseSpec(kind="IFFT1D"),
            projection=ProjectionSpec(domain="REAL_VECTOR"),
            quality=ReconstructQuality(reconstructable=True, error_metric="mse", max_error=1e-9),
            confidence=1.0
        )

    def project(self) -> list[float]:
        """
        Project the atom back to the real domain using the inverse transform.
        """
        import numpy as np
        
        if self.inverse.kind != "IFFT1D":
             raise NotImplementedError(f"Inverse transform {self.inverse.kind} not supported in v0.1")

        if not self.repr:
            return []

        # 1. Reconstruct spectrum
        spectrum = np.array([c.to_complex() for c in self.repr])
        
        # 2. IFFT
        reconstructed = np.fft.ifft(spectrum)
        
        # 3. Take real part (assuming hermitian symmetry or ignoring imaginary error)
        # For v0.1 we just take real.
        return reconstructed.real.tolist()
