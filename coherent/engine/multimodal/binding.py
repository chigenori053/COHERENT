
"""
Holographic Binding Mechanism.
Implements vector symbolic architectures (VSA) operations: Bind and Unbind.
Uses Circular Convolution for Binding and Circular Correlation for Unbinding.
"""

from __future__ import annotations
import torch
from coherent.engine.holographic.data_types import HolographicTensor, SpectrumConfig

class HolographicBinding:
    """
    Binding mechanism for Holographic Tensors.
    Combines two complex tensors into a single tensor that preserves structural information.
    
    Operation:
    - Binding: Element-wise Multiplication in Frequency Domain (Circular Convolution in Time Domain).
      Result = A * B
    - Unbinding: Element-wise Multiplication by Conjugate (Circular Correlation).
      Result = C * A.conj() -> Approximation of B
    """
    
    @staticmethod
    def bind(tensor_a: HolographicTensor, tensor_b: HolographicTensor) -> HolographicTensor:
        """
        Binds two holographic tensors.
        Args:
            tensor_a: Complex spectrum A [Dim]
            tensor_b: Complex spectrum B [Dim]
        Returns:
            Bound complex spectrum [Dim]
        """
        # Ensure dimensionality match
        if tensor_a.tensor.shape != tensor_b.tensor.shape:
             raise ValueError(f"Shape mismatch: {tensor_a.tensor.shape} vs {tensor_b.tensor.shape}")
             
        # Element-wise multiplication in frequency domain
        # Corresponds to circular convolution in spatial domain
        # This is a standard VSA binding operation (e.g. HRR)
        bound = tensor_a.tensor * tensor_b.tensor
        
        # Normalize to maintain unit energy (optional but good for stability)
        # bound = bound / (bound.abs().max() + 1e-8)
        
        return HolographicTensor(bound)

    @staticmethod
    def unbind(bound_tensor: HolographicTensor, key_tensor: HolographicTensor) -> HolographicTensor:
        """
        Retrieves original tensor from bound tensor using the key.
        Args:
            bound_tensor: The result of bind(A, B)
            key_tensor: One of the operands (e.g. A)
        Returns:
            Approximation of the other operand (e.g. B)
        """
        if bound_tensor.tensor.shape != key_tensor.tensor.shape:
             raise ValueError(f"Shape mismatch: {bound_tensor.tensor.shape} vs {key_tensor.tensor.shape}")
             
        # Unbinding: Multiplication by complex conjugate
        # In HRR/VSA, this is the approximate inverse
        # C = A * B -> C * A' approx B
        retrieved = bound_tensor.tensor * key_tensor.tensor.conj()
        
        return HolographicTensor(retrieved)

    @staticmethod
    def bundle(tensors: list[HolographicTensor]) -> HolographicTensor:
        """
        Superposition (Add) multiple tensors.
        """
        if not tensors:
            # Return zero tensor of default dim? Need context or explicit dim.
            raise ValueError("Cannot bundle empty list")
            
        dim = tensors[0].tensor.shape[0]
        result = torch.zeros(dim, dtype=torch.complex64)
        
        for t in tensors:
             result += t.tensor
             
        # Normalize?
        # result = result / len(tensors)
             
        return HolographicTensor(result)
