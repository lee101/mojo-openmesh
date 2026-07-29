from __future__ import annotations

import importlib
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_ROOT = os.path.join(ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import mojo_openmesh as mom


def load_upstream():
    original = list(sys.path)
    try:
        sys.path[:] = [
            path
            for path in sys.path
            if os.path.abspath(path or os.curdir) != os.path.abspath(PYTHON_ROOT)
        ]
        sys.modules.pop("openmesh", None)
        return importlib.import_module("openmesh")
    finally:
        sys.path[:] = original


upstream = load_upstream()


def grid(n):
    x, y = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64)
    ids = np.arange((n + 1) ** 2, dtype=np.int32).reshape(n + 1, n + 1)
    a = ids[:-1, :-1].ravel()
    b = ids[:-1, 1:].ravel()
    c = ids[1:, 1:].ravel()
    d = ids[1:, :-1].ravel()
    faces = np.empty((2 * n * n, 3), dtype=np.int32)
    faces[0::2] = np.column_stack((a, b, c))
    faces[1::2] = np.column_stack((a, c, d))
    return points, faces


def make(module, points, faces):
    mesh = module.TriMesh()
    mesh.add_vertices(points)
    mesh.add_faces(faces)
    return mesh


def timeit(fn, repeat=5):
    best = math.inf
    value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def time_prepared(factory, operation, repeat=3):
    best = math.inf
    for _ in range(repeat):
        mesh = factory()
        start = time.perf_counter()
        operation(mesh)
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def duration(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def main():
    points, faces = grid(300)
    ours = make(mom, points, faces)
    theirs = make(upstream, points, faces)
    rows = []

    mojo_t, _ = timeit(lambda: make(mom, points, faces), repeat=4)
    upstream_t, _ = timeit(lambda: make(upstream, points, faces), repeat=4)
    rows.append(("build topology (180k faces)", mojo_t, upstream_t))

    mojo_t, ours_vv = timeit(ours.vertex_vertex_indices)
    upstream_t, theirs_vv = timeit(theirs.vertex_vertex_indices)
    assert ours_vv.shape == theirs_vv.shape
    rows.append(("all vertex one-rings (90.6k)", mojo_t, upstream_t))

    mojo_t, ours_lengths = timeit(ours.calc_edge_lengths)
    upstream_t, theirs_lengths = timeit(
        lambda: np.fromiter(
            (theirs.calc_edge_length(edge) for edge in theirs.edges()),
            dtype=np.float64,
            count=theirs.n_edges(),
        ),
        repeat=3,
    )
    np.testing.assert_allclose(ours_lengths, theirs_lengths)
    rows.append(("all edge lengths (270.6k)", mojo_t, upstream_t))

    theirs.request_face_normals()
    mojo_t, ours_normals = timeit(ours.calc_face_normals)

    def upstream_normals():
        theirs.update_face_normals()
        return np.asarray(theirs.face_normals()).copy()

    upstream_t, theirs_normals = timeit(upstream_normals)
    np.testing.assert_allclose(ours_normals, theirs_normals)
    rows.append(("all face normals (180k)", mojo_t, upstream_t))

    small_points, small_faces = grid(160)
    center = 80 * 161 + 80
    diagonal = center + 162

    def flip(mesh):
        halfedge = mesh.find_halfedge(
            mesh.vertex_handle(center), mesh.vertex_handle(diagonal)
        )
        mesh.flip(mesh.edge_handle(halfedge))

    mojo_t = time_prepared(lambda: make(mom, small_points, small_faces), flip)
    upstream_t = time_prepared(lambda: make(upstream, small_points, small_faces), flip)
    rows.append(("single flip (51.2k-face mesh)", mojo_t, upstream_t))

    print(f"Machine: {cpu_name()} | {platform.system()} {platform.machine()} | Python {platform.python_version()}")
    print()
    print("| operation | mojo-openmesh | upstream OpenMesh | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_t, upstream_t in rows:
        ratio = upstream_t / mojo_t
        ratio_text = f"{ratio:.4f}x" if ratio < 0.01 else f"{ratio:.2f}x"
        print(
            f"| {name} | {duration(mojo_t)} | {duration(upstream_t)} | "
            f"{ratio_text} |"
        )


if __name__ == "__main__":
    main()
