"""BTI-Net."""
import os as _os

_ROOT = _os.environ.get("BTINET_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("HF_HOME", _os.path.join(_ROOT, ".cache", "huggingface"))
_os.environ.setdefault("TORCH_HOME", _os.path.join(_ROOT, ".cache", "torch"))
