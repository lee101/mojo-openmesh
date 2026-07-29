# mojo-openmesh

`mojo-openmesh` is a Mojo implementation of the compute-heavy core of
[OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/): triangular
half-edge topology construction, one-ring traversal, geometry kernels, and edge
collapse, split, and flip. Its Python package is also named `openmesh`, and the
covered API uses OpenMesh's handle classes, method names, signatures, indexing,
and explicit boundary halfedges.

This is a focused port, not bindings to the C++ library. NumPy owns the mesh
storage and calls one compiled Mojo shared library through `ctypes`.

## Covered API

- `TriMesh`, `VertexHandle`, `HalfedgeHandle`, `EdgeHandle`, and `FaceHandle`
- vertex/face insertion, counts, iterators, points, and face index arrays
- paired halfedges (`2 * edge`, `2 * edge + 1`) and explicit linked boundary loops
- `vv`, `voh`, `vih`, `ve`, `vf`, `fv`, `fh`, `fe`, and `ff` circulators
- halfedge navigation, handle conversion, boundary tests, `find_halfedge`, and
  padded vector index APIs such as `vertex_vertex_indices()`
- all-edge lengths, normalized face normals, edge vectors, and face centroids
- `is_flip_ok`/`flip`, edge `split`, and
  `is_collapse_ok`/directed-halfedge `collapse`

The test suite compares this API directly with conda-forge's real
`openmesh-python 1.2.1` package, which wraps OpenMesh C++.

Not covered are polygonal `PolyMesh`, file I/O, arbitrary properties, colors and
texture coordinates, deletion/status workflows, normal property management,
decimation modules, or OpenMesh's lower-level connectivity mutation methods.
Split and collapse compact and rebuild connectivity, so existing edge,
halfedge, and face handles must be treated as invalid after either operation.
Flip preserves handle indices but changes the affected connectivity.
`garbage_collection()` is a compatibility no-op.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-openmesh.so`. `pixi run test` also installs and
tests against the upstream Python binding.

## Usage

```python
import numpy as np
import openmesh as om

mesh = om.TriMesh()
mesh.add_vertices(np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
]))
mesh.add_faces(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))

diagonal = mesh.find_halfedge(mesh.vertex_handle(0), mesh.vertex_handle(2))
edge = mesh.edge_handle(diagonal)
assert mesh.is_flip_ok(edge)
mesh.flip(edge)

print(mesh.face_vertex_indices())
# [[1 2 3]
#  [3 0 1]]
```

The documented point overload also works:

```python
edge = next(mesh.edges())
new_vertex = mesh.split(edge, np.array([0.5, 0.5, 0.0]))
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14:

| operation | mojo-openmesh | upstream OpenMesh | upstream / Mojo |
| --- | ---: | ---: | ---: |
| build topology (180k faces) | 97.54 ms | 97.62 ms | 1.00x |
| all vertex one-rings (90.6k) | 950.0 us | 5.09 ms | 5.36x |
| all edge lengths (270.6k) | 830.9 us | 351.00 ms | 422.42x |
| all face normals (180k) | 1.68 ms | 5.76 ms | 3.42x |
| single flip (51.2k-face mesh) | 131.8 us | 28.3 us | 0.21x |

The edge-length result includes Python API overhead: upstream exposes only a
per-edge call, while this port crosses the FFI once for the whole array. Each
reported time is the best observed duration from the repetitions in
`bench/bench.py`; the benchmark validates result parity before printing. A flip
updates its two faces and six affected halfedges in place, then rebuilds
one-ring CSR only if a later query needs it. No CPU parallel or GPU path is
implemented.

## How it works

Points are contiguous `float64[n, 3]` and faces are contiguous `int64[m, 3]`
NumPy arrays. Mojo builds structure-of-arrays halfedge storage: from/to vertex,
next/previous, opposite, face, and edge arrays. Every edge owns two adjacent
halfedge slots, including a boundary slot when only one incident face exists.
An open-addressed edge table makes construction linear in the number of faces,
then a second pass links boundary loops and emits compact vertex one-ring CSR
storage.

NumPy allocates and owns every input, output, hash, and scratch buffer. The C ABI
receives buffers as integer addresses, reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` inside non-parametric
`@export(...) ... abi("C")` functions, and never retains or frees Python
memory. All kernels live in one Mojo compilation unit to keep build overhead
fixed.

Split and collapse kernels rewrite oriented face triples in Mojo and rebuild
through the validating topology builder. Flip validation and connectivity
updates are local; its one-ring arrays are refreshed lazily after the edit.

## License

MIT
