import pytest
import torch
import numpy as np
from causalscript.core.optical.layer import OpticalScoringLayer

def test_optical_layer_initialization():
    layer = OpticalScoringLayer(input_dim=10, output_dim=5)
    assert layer.input_dim == 10
    assert layer.output_dim == 5
    assert layer.weights.shape == (5, 10)
    assert layer.weights.dtype == torch.cfloat
    assert layer.weights.requires_grad

def test_optical_layer_forward_shape():
    layer = OpticalScoringLayer(input_dim=10, output_dim=5)
    # Batch size 1
    input_tensor = torch.randn(10)
    intensity, ambiguity = layer(input_tensor)
    
    assert intensity.shape == (1, 5)
    # Intensity should be real and non-negative
    assert intensity.dtype == torch.float32 or intensity.dtype == torch.float64
    assert torch.all(intensity >= 0)
    
    assert isinstance(ambiguity, float)
    assert 0.0 <= ambiguity <= 1.0

def test_optical_layer_forward_batch():
    layer = OpticalScoringLayer(input_dim=10, output_dim=5)
    # Batch size 3
    input_tensor = torch.randn(3, 10)
    intensity, ambiguity = layer(input_tensor)
    
    assert intensity.shape == (3, 5)
    # Ambiguity is currently calc based on [0] for single inference compatibility, 
    # ensuring it doesn't crash on batch is the goal here.
    assert isinstance(ambiguity, float)

def test_optical_layer_backward():
    layer = OpticalScoringLayer(input_dim=10, output_dim=5)
    input_tensor = torch.randn(10, dtype=torch.cfloat)
    
    # Forward
    intensity, _ = layer(input_tensor)
    
    # Loss (minimize sum)
    loss = intensity.sum()
    
    # Backward
    layer.zero_grad()
    loss.backward()
    
    assert layer.weights.grad is not None
    # Gradient should be complex since weights are complex
    assert layer.weights.grad.dtype == torch.cfloat
