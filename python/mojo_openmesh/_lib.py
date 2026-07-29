from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_OPENMESH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-openmesh.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mom_build_topology": ([I] * 15, I),
    "mom_vertex_rings": ([I] * 9, I),
    "mom_padded_rows": ([I] * 5, None),
    "mom_edge_lengths": ([I] * 5, None),
    "mom_face_normals": ([I] * 4, None),
    "mom_flip_edge": ([I] * 11, I),
    "mom_split_edge": ([I] * 10, I),
    "mom_collapse_halfedge": ([I] * 6, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "openmesh.mojo")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError(f"unsupported FFI buffer dtype: {array.dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    if not array.flags.writeable:
        raise ValueError("FFI buffers must be writable")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)


def i64(values=(), *, shape=None) -> np.ndarray:
    if shape is not None:
        return np.empty(shape, dtype=np.int64)
    return np.ascontiguousarray(values, dtype=np.int64)


def f64(values=(), *, shape=None) -> np.ndarray:
    if shape is not None:
        return np.empty(shape, dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)
