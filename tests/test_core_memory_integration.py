
import pytest
import torch
import numpy as np
from coherent.core.memory.optical_store import OpticalFrequencyStore

def test_optical_store_with_crs_atoms():
    # 1. Initialize Store
    # Dim 4 to match our simple test vectors
    store = OpticalFrequencyStore(vector_dim=4, capacity=10)
    
    # 2. Add Data
    # Vector A: [1, 0, 1, 0] -> Periodic
    # Vector B: [0, 1, 0, 1] -> Periodic shifted
    vec_a = [1.0, 0.0, 1.0, 0.0]
    vec_b = [0.0, 1.0, 0.0, 1.0]
    
    store.add(
        collection_name="test",
        vectors=[vec_a, vec_b],
        metadatas=[{"label": "A"}, {"label": "B"}],
        ids=["id_a", "id_b"]
    )
    
    # 3. Verify Internal Storage is Complex (Spectrum)
    # The store uses MemoryAtom.from_real_vector which uses FFT.
    # FFT of [1,0,1,0] is [2, 0, 2, 0] (real) ? No.
    # np.fft.fft([1,0,1,0]) -> [2.+0.j, 0.+0.j, 2.+0.j, 0.+0.j] (Actually it is real because symmetric/periodic)
    
    # Let's try [1, 2, 3, 4] for a non-trivial spectrum
    vec_c = [1.0, 2.0, 3.0, 4.0]
    store.add(
        collection_name="test",
        vectors=[vec_c],
        metadatas=[{"label": "C"}],
        ids=["id_c"]
    )
    
    # Index 2 should correspond to vec_c
    stored_c = store.optical_layer.optical_memory.data[2]
    assert stored_c.is_complex()
    
    # 4. Query (Associative Recall)
    # Query with A should return A
    results = store.query("test", vec_a, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "id_a"
    assert results[0]["score"] > 0.9 # High resonance
    
    # Query with something similar to C
    # [1.1, 2.0, 3.0, 4.0]
    vec_c_prime = [1.1, 2.0, 3.0, 4.0]
    results_c = store.query("test", vec_c_prime, top_k=1)
    assert results_c[0]["id"] == "id_c"

def test_optical_store_capacity():
    store = OpticalFrequencyStore(vector_dim=2, capacity=2)
    store.add("t", [[1,0]], [{}], ["1"])
    store.add("t", [[0,1]], [{}], ["2"])
    
    with pytest.raises(MemoryError):
        store.add("t", [[1,1]], [{}], ["3"])
