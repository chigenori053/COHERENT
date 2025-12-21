import pytest
from coherent.engine.language.decomposer import Decomposer

class TestDecomposer:
    def setup_method(self):
        self.decomposer = Decomposer()

    def test_explicit_delimiters(self):
        text = "x = 1; y = 2\nz = 3"
        segments = self.decomposer.decompose(text)
        assert segments == ["x = 1", "y = 2", "z = 3"]

    def test_japanese_soft_delimiter(self):
        text = "x = 1の場合 y = 2"
        segments = self.decomposer.decompose(text)
        # Should split at "の場合" and ideally remove it or handle it?
        # For now, let's assume it splits and cleans up.
        assert "x = 1" in segments[0]
        assert "y = 2" in segments[1]
        assert len(segments) == 2

    def test_implicit_variable_boundary(self):
        # The core faulty case: "y = 3x + 2 x = 1"
        text = "y = 3x + 2 x = 1"
        segments = self.decomposer.decompose(text)
        assert segments == ["y = 3x + 2", "x = 1"]

    def test_complex_scenario(self):
        text = "y = 3x + 2 x = 1の場合 y = 5 x = -2の場合 y = -4"
        segments = self.decomposer.decompose(text)
        expected = [
            "y = 3x + 2",
            "x = 1",
            "y = 5",
            "x = -2",
            "y = -4"
        ]
        assert segments == expected

    def test_no_split_needed(self):
        text = "Solve x^2 + 2x + 1 = 0"
        segments = self.decomposer.decompose(text)
        assert segments == ["Solve x^2 + 2x + 1 = 0"]

    def test_multiple_implicit(self):
        # "y = 5 x = -2"
        text = "y = 5 x = -2"
        segments = self.decomposer.decompose(text)
        assert segments == ["y = 5", "x = -2"]
