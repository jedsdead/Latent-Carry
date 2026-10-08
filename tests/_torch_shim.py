"""A numpy-backed stand-in for the handful of torch operations the carry path uses.

The suite keeps its "numpy only, no torch, no GPU" property.  This implements
exactly what patches.py touches on the capture, fingerprint, alignment,
restore and snapshot paths - no more - with torch's semantics for each.  If
patches.py starts using an operation that is not here, the test fails with an
AttributeError naming it, rather than passing on a silent approximation.
"""

import sys
import types

import numpy as np

float32 = np.dtype(np.float32)
float16 = np.dtype(np.float16)
uint8 = np.dtype(np.uint8)


def _unwrap(value):
    return value.a if isinstance(value, Tensor) else value


def _dim(dim):
    return tuple(dim) if isinstance(dim, (list, tuple)) else dim


class Tensor:
    def __init__(self, array):
        self.a = np.asarray(array)

    # --- shape and type
    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return self.a.ndim

    @property
    def dtype(self):
        return self.a.dtype

    @property
    def device(self):
        return "cpu"

    def numel(self):
        return int(self.a.size)

    def __len__(self):
        return len(self.a)

    # --- conversions (all copy, like torch's out-of-place ops)
    def float(self):
        return Tensor(self.a.astype(np.float32))

    def to(self, *args, dtype=None, **kwargs):
        target = dtype if dtype is not None else next(
            (a for a in args if isinstance(a, np.dtype)), None)
        return Tensor(self.a.astype(target)) if target is not None else Tensor(self.a.copy())

    def cpu(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.a.copy())

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.a))

    def numpy(self):
        return self.a

    def __float__(self):
        return float(self.a)

    def __int__(self):
        return int(self.a)

    def __index__(self):
        return int(self.a)

    # --- indexing
    def __getitem__(self, key):
        return Tensor(self.a[key])

    def __setitem__(self, key, value):
        self.a[key] = _unwrap(value)

    # --- arithmetic
    def __sub__(self, other):
        return Tensor(self.a - _unwrap(other))

    def __add__(self, other):
        return Tensor(self.a + _unwrap(other))

    def __mul__(self, other):
        return Tensor(self.a * _unwrap(other))

    def __truediv__(self, other):
        return Tensor(self.a / _unwrap(other))

    def __rsub__(self, other):
        return Tensor(_unwrap(other) - self.a)

    def __radd__(self, other):
        return Tensor(_unwrap(other) + self.a)

    def __rmul__(self, other):
        return Tensor(_unwrap(other) * self.a)

    def __neg__(self):
        return Tensor(-self.a)

    def add(self, value):
        return Tensor(self.a + value)

    def add_(self, value):
        self.a = self.a + _unwrap(value)
        return self

    def sub(self, value):
        return Tensor(self.a - value)

    def sub_(self, value):
        self.a = self.a - _unwrap(value)
        return self

    def div(self, value):
        return Tensor(self.a / value)

    def div_(self, value):
        self.a = self.a / _unwrap(value)
        return self

    def mul(self, value):
        return Tensor(self.a * value)

    def abs(self):
        return Tensor(np.abs(self.a))

    def pow(self, exponent):
        return Tensor(self.a ** exponent)

    def sqrt(self):
        return Tensor(np.sqrt(self.a))

    def clamp(self, low=None, high=None):
        return Tensor(np.clip(self.a, low, high))

    def clamp_min(self, low):
        return Tensor(np.maximum(self.a, low))

    def round(self):
        return Tensor(np.round(self.a))

    def round_(self):
        self.a = np.round(self.a)
        return self

    def clamp_(self, low=None, high=None):
        self.a = np.clip(self.a, low, high)
        return self

    def mul_(self, value):
        self.a = self.a * _unwrap(value)
        return self

    def view(self, *shape):
        return Tensor(self.a.reshape(*shape))

    # --- reductions
    def mean(self, dim=None, keepdim=False):
        return Tensor(self.a.mean(axis=_dim(dim), keepdims=keepdim))

    def std(self, dim=None, keepdim=False):
        return Tensor(self.a.std(axis=_dim(dim), keepdims=keepdim, ddof=1))

    def max(self):
        return Tensor(self.a.max())

    def min(self):
        return Tensor(self.a.min())

    def argmin(self):
        return Tensor(self.a.argmin())

    def argmax(self):
        return Tensor(self.a.argmax())

    # --- reshaping
    def reshape(self, *shape):
        return Tensor(self.a.reshape(*shape))

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.a, dim))

    def permute(self, *dims):
        return Tensor(self.a.transpose(*dims))

    def __bool__(self):
        # torch raises for multi-element tensors; so does this.
        if self.a.size != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return bool(self.a)


def tensor(data, dtype=None):
    return Tensor(np.asarray(data, dtype=dtype))


def from_numpy(array):
    return Tensor(array)


def stack(tensors, dim=0):
    return Tensor(np.stack([_unwrap(t) for t in tensors], axis=dim))


def cat(tensors, dim=0):
    return Tensor(np.concatenate([_unwrap(t) for t in tensors], axis=dim))


def as_tensor(data, dtype=None, device=None):
    return Tensor(np.asarray(_unwrap(data), dtype=dtype))


def einsum(spec, *operands):
    return Tensor(np.einsum(spec, *[_unwrap(o) for o in operands]))


def zeros(*shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype))


def ones(*shape, dtype=np.float32):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape, dtype=dtype))


def zeros_like(value):
    return Tensor(np.zeros_like(_unwrap(value)))


def is_tensor(value):
    return isinstance(value, Tensor)


def _adaptive_avg_pool2d(value, size):
    """Exact for input sizes divisible by the output size, which is all the tests use."""
    array = _unwrap(value)
    out_h, out_w = size
    height, width = array.shape[-2:]
    if height % out_h or width % out_w:
        raise ValueError("shim adaptive_avg_pool2d needs divisible sizes")
    shaped = array.reshape(*array.shape[:-2], out_h, height // out_h, out_w, width // out_w)
    return Tensor(shaped.mean(axis=(-3, -1)))


class inference_mode:
    """Context manager, as torch.inference_mode(); nothing to do on numpy."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install():
    """Register as `torch` in sys.modules.  Returns the module."""
    module = types.ModuleType("torch")
    for name in ("Tensor", "tensor", "from_numpy", "stack", "cat", "zeros", "ones", "zeros_like", "as_tensor", "einsum",
                 "is_tensor", "float32", "float16", "uint8", "inference_mode"):
        setattr(module, name, globals()[name])
    module.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(adaptive_avg_pool2d=_adaptive_avg_pool2d))
    sys.modules["torch"] = module
    return module
