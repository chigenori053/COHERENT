import os

# Point package imports to the legacy Python package root now under coherent/core,
# while keeping coherent/apps etc. discoverable.
_ROOT = os.path.dirname(__file__)
_LEGACY_ROOT = os.path.join(_ROOT, "Core")
__path__ = [_LEGACY_ROOT, _ROOT]
__version__ = "0.1.0"
