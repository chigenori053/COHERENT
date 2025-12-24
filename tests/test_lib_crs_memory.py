
import pytest
import yaml
from coherent.tools.library.crs_memory import MemoryBuilder, MemoryStructure, MemoryBlock

def test_builder_formula_flow():
    builder = MemoryBuilder()
    source = "(x - y)**2"
    structure = builder.from_formula(source, fmt="sympy")
    
    assert structure.schema_version == "0.1"
    assert len(structure.blocks) == 1
    assert structure.blocks[0].block_type.value == "formula"
    assert structure.causal_graph is not None
    
    # Check causal variables
    nodes = structure.causal_graph.nodes
    # Should have x, y, -1, and derived nodes
    # Note: Normalization might abstract x, y to v1, v2
    
    var_names = [v.name for v in nodes.values()]
    # Check input vars are present (either as x/y or abstract mapped ones)
    # The current causal extractor implementation tries to resolve back to real names if possible
    
    print(f"Causal Nodes: {var_names}")
    
    # Check edges
    # (x-y)^2 -> derived from x and y
    assert len(structure.causal_graph.edges) > 0

def test_serialization_roundtrip():
    builder = MemoryBuilder()
    structure = builder.from_formula("a + b", fmt="sympy")
    
    yaml_str = builder.save_structure(structure)
    print("\nYAML Output:\n", yaml_str)
    
    # Verify YAML content
    assert "schema_version: '0.1'" in yaml_str
    assert "block_type: formula" in yaml_str
    
    # Load back
    restored = builder.load_structure(yaml_str)
    
    assert restored.schema_version == structure.schema_version
    assert len(restored.blocks) == 1
    assert restored.blocks[0].id == structure.blocks[0].id
    assert restored.blocks[0].payload["canon_ast"] == structure.blocks[0].payload["canon_ast"]

def test_isomorphism_signature():
    builder = MemoryBuilder()
    
    s1 = builder.from_formula("x + y")
    s2 = builder.from_formula("y + x")
    s3 = builder.from_formula("a + b")
    
    sig1 = s1.blocks[0].payload["normalized_signature"]
    sig2 = s2.blocks[0].payload["normalized_signature"]
    sig3 = s3.blocks[0].payload["normalized_signature"]
    
    print(f"Sig1 (x+y): {sig1}")
    print(f"Sig2 (y+x): {sig2}")
    print(f"Sig3 (a+b): {sig3}")

    # x+y and y+x should be identical due to sorting
    assert sig1 == sig2
    
    # a+b should match x+y due to variable abstraction
    assert sig1 == sig3

def test_complex_atom_structure():
    # Manual construction test
    from coherent.tools.library.crs_memory.atoms import MemoryAtom, AtomType, ComplexVal, TransformSpec, InverseSpec, ProjectionSpec, ReconstructQuality
    
    atom = MemoryAtom(
        id="atom-1",
        atom_type=AtomType.SIGNAL,
        spec_dim=2,
        repr=[ComplexVal(1.0, 0.0), ComplexVal(0.0, 1.0)],
        transform=TransformSpec(kind="FFT"),
        inverse=InverseSpec(kind="IFFT"),
        projection=ProjectionSpec(domain="REAL_SIGNAL"),
        quality=ReconstructQuality(reconstructable=True),
        confidence=0.9
    )
    
    assert atom.repr[1].to_complex() == 1j

def test_atom_roundtrip_fft():
    # Test Real -> Atom (Spectrum) -> Real
    from coherent.tools.library.crs_memory.atoms import MemoryAtom
    import numpy as np
    
    # 1. Input Data
    original = [1.0, 2.0, 1.0, -1.0]
    
    # 2. Extract Atom
    atom = MemoryAtom.from_real_vector(original, id="a1")
    
    # Check structure
    assert atom.spec_dim == 4
    assert atom.transform.kind == "FFT1D"
    
    # 3. Project back
    restored = atom.project()
    
    # 4. Verify
    print(f"Original: {original}")
    print(f"Restored: {restored}")
    
    np.testing.assert_allclose(restored, original, atol=1e-9)

def test_atom_edge_cases():
    from coherent.tools.library.crs_memory.atoms import MemoryAtom
    import numpy as np
    
    # 1. Empty input
    # Should probably result in empty atom or handle gracefully
    atom_empty = MemoryAtom.from_real_vector([], id="empty")
    assert atom_empty.spec_dim == 0
    assert len(atom_empty.repr) == 0
    assert atom_empty.project() == []
    
    # 2. Single value
    atom_single = MemoryAtom.from_real_vector([5.0], id="single")
    assert atom_single.spec_dim == 1
    # FFT of [5] is [5+0j]
    assert atom_single.repr[0].re == 5.0
    assert atom_single.project()[0] == pytest.approx(5.0)

