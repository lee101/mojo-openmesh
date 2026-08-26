from __future__ import annotations

from collections.abc import Iterator
import operator

import numpy as np

from ._lib import addr, f64, i64, lib
from .handles import EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle


class TopologyError(ValueError):
    pass


def _idx(handle, cls) -> int:
    if not isinstance(handle, cls):
        raise TypeError(f"expected {cls.__name__}, got {type(handle).__name__}")
    return handle.idx()


def _dense(rows: list[np.ndarray], dtype=np.int32) -> np.ndarray:
    width = max((len(row) for row in rows), default=0)
    result = np.full((len(rows), width), -1, dtype=dtype)
    for index, row in enumerate(rows):
        result[index, : len(row)] = row
    return result


def _face_array(values) -> np.ndarray:
    faces = np.asarray(values)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("face_vertex_indices must have shape (n, 3)")
    if faces.dtype.kind not in "iu":
        raise TypeError("face_vertex_indices must contain integers")
    if faces.size:
        info = np.iinfo(np.int64)
        minimum = int(faces.min())
        maximum = int(faces.max())
        if minimum < info.min or maximum > info.max:
            raise OverflowError("face vertex index does not fit in int64")
    return np.array(faces, dtype=np.int64, order="C", copy=True)


def _vertex_index(value) -> int:
    if isinstance(value, VertexHandle):
        return value.idx()
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError("face vertices must be VertexHandle or integer") from None


class TriMesh:
    InvalidVertexHandle = VertexHandle(-1)
    InvalidHalfedgeHandle = HalfedgeHandle(-1)
    InvalidEdgeHandle = EdgeHandle(-1)
    InvalidFaceHandle = FaceHandle(-1)

    def __init__(self, points=None, face_vertex_indices=None):
        self._points = np.empty((0, 3), dtype=np.float64)
        self._faces = np.empty((0, 3), dtype=np.int64)
        self._dirty = True
        self._rings_dirty = True
        self._edge_count = 0
        if points is not None:
            self.add_vertices(points)
        if face_vertex_indices is not None:
            self.add_faces(face_vertex_indices)

    def _invalidate(self) -> None:
        self._dirty = True
        self._rings_dirty = True

    def _ensure_topology(self) -> None:
        if not self._dirty:
            return
        nv, nf = len(self._points), len(self._faces)
        if nf and not nv:
            raise TopologyError("face has an invalid or repeated vertex")
        if nf == 0:
            self._edge_count = 0
            self._he_from = np.empty(0, dtype=np.int64)
            self._he_to = np.empty(0, dtype=np.int64)
            self._he_next = np.empty(0, dtype=np.int64)
            self._he_prev = np.empty(0, dtype=np.int64)
            self._he_opp = np.empty(0, dtype=np.int64)
            self._he_face = np.empty(0, dtype=np.int64)
            self._he_edge = np.empty(0, dtype=np.int64)
            self._face_he = np.empty((0, 3), dtype=np.int64)
            self._vertex_out = np.full(nv, -1, dtype=np.int64)
            self._ring_offsets = np.zeros(nv + 1, dtype=np.int64)
            self._ring_halfedges = np.empty(0, dtype=np.int64)
            self._ring_neighbors = np.empty(0, dtype=np.int64)
            self._flip_buffers = ()
            self._dirty = False
            self._rings_dirty = False
            return

        max_halfedges = 6 * nf
        capacity = 8
        while capacity < 4 * nf:
            capacity *= 2
        hash_keys = i64(shape=capacity)
        hash_values = i64(shape=capacity)
        arrays = [i64(shape=max_halfedges) for _ in range(7)]
        face_he = i64(shape=(nf, 3))
        vertex_out = i64(shape=nv)
        result = lib().mom_build_topology(
            nv,
            nf,
            addr(self._faces),
            addr(hash_keys),
            addr(hash_values),
            capacity,
            *(addr(array) for array in arrays),
            addr(face_he),
            addr(vertex_out),
        )
        messages = {
            -1: "face has an invalid or repeated vertex",
            -2: "topology hash table overflow",
            -3: "duplicate orientation or non-manifold edge",
            -4: "non-manifold boundary vertex",
        }
        if result < 0:
            raise TopologyError(messages.get(result, "invalid topology"))

        self._edge_count = result
        nh = 2 * result
        (
            self._he_from,
            self._he_to,
            self._he_next,
            self._he_prev,
            self._he_opp,
            self._he_face,
            self._he_edge,
        ) = (array[:nh] for array in arrays)
        self._face_he = face_he
        self._vertex_out = vertex_out
        # Keep the owning arrays, not bare addresses, alive between calls.
        self._flip_buffers = (
            self._faces,
            self._he_from,
            self._he_to,
            self._he_next,
            self._he_prev,
            self._he_opp,
            self._he_face,
            self._face_he,
            self._vertex_out,
        )
        self._flip_addresses = i64(tuple(addr(array) for array in self._flip_buffers))
        self._flip_address = addr(self._flip_addresses)
        self._flip_kernel = lib().mom_flip_edge
        self._dirty = False
        self._rings_dirty = True
        self._ensure_rings()

    def _ensure_rings(self) -> None:
        self._ensure_topology()
        if not self._rings_dirty:
            return
        nv = self.n_vertices()
        nh = self.n_halfedges()
        offsets = i64(shape=nv + 1)
        ring_halfedges = i64(shape=nh)
        ring_neighbors = i64(shape=nh)
        total = lib().mom_vertex_rings(
            nv,
            nh,
            addr(self._vertex_out),
            addr(self._he_to),
            addr(self._he_next),
            addr(self._he_opp),
            addr(offsets),
            addr(ring_halfedges),
            addr(ring_neighbors),
        )
        if total != nh:
            raise TopologyError("vertex one-ring is disconnected")
        self._ring_offsets = offsets
        self._ring_halfedges = ring_halfedges
        self._ring_neighbors = ring_neighbors
        self._rings_dirty = False

    def _ring(self, vh: VertexHandle) -> np.ndarray:
        self._ensure_rings()
        v = _idx(vh, VertexHandle)
        if not 0 <= v < len(self._points):
            return np.empty(0, dtype=np.int64)
        start, stop = self._ring_offsets[v : v + 2]
        return self._ring_halfedges[start:stop]

    def _replace_faces(self, faces: np.ndarray) -> None:
        old = self._faces
        self._faces = np.ascontiguousarray(faces, dtype=np.int64)
        self._invalidate()
        try:
            self._ensure_topology()
        except Exception:
            self._faces = old
            self._invalidate()
            raise

    def add_vertex(self, point) -> VertexHandle:
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("a vertex point must have shape (3,)")
        index = len(self._points)
        self._points = np.vstack((self._points, point))
        self._invalidate()
        return VertexHandle(index)

    def add_vertices(self, points) -> None:
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if len(self._points):
            self._points = np.vstack((self._points, points))
        else:
            self._points = points.copy()
        self._invalidate()

    def add_face(self, *args) -> FaceHandle:
        vertices = args[0] if len(args) == 1 else args
        vertices = list(vertices)
        if len(vertices) != 3:
            raise ValueError("TriMesh faces need exactly three vertices")
        row = np.array(
            [_vertex_index(v) for v in vertices],
            dtype=np.int64,
        )
        index = len(self._faces)
        candidate = np.vstack((self._faces, row))
        try:
            self._replace_faces(candidate)
        except TopologyError:
            return FaceHandle(-1)
        return FaceHandle(index)

    def add_faces(self, face_vertex_indices) -> None:
        converted = _face_array(face_vertex_indices)
        candidate = converted if not len(self._faces) else np.vstack((self._faces, converted))
        self._replace_faces(candidate)

    def clear(self) -> None:
        self._points = np.empty((0, 3), dtype=np.float64)
        self._faces = np.empty((0, 3), dtype=np.int64)
        self._invalidate()

    clean = clear

    def reserve(self, *args) -> None:
        pass

    def n_vertices(self) -> int:
        return len(self._points)

    def n_faces(self) -> int:
        return len(self._faces)

    def n_edges(self) -> int:
        self._ensure_topology()
        return self._edge_count

    def n_halfedges(self) -> int:
        return 2 * self.n_edges()

    def vertices_empty(self) -> bool:
        return self.n_vertices() == 0

    def faces_empty(self) -> bool:
        return self.n_faces() == 0

    def edges_empty(self) -> bool:
        return self.n_edges() == 0

    def halfedges_empty(self) -> bool:
        return self.n_halfedges() == 0

    def vertices(self) -> Iterator[VertexHandle]:
        return iter(VertexHandle(i) for i in range(self.n_vertices()))

    def faces(self) -> Iterator[FaceHandle]:
        return iter(FaceHandle(i) for i in range(self.n_faces()))

    def edges(self) -> Iterator[EdgeHandle]:
        return iter(EdgeHandle(i) for i in range(self.n_edges()))

    def halfedges(self) -> Iterator[HalfedgeHandle]:
        return iter(HalfedgeHandle(i) for i in range(self.n_halfedges()))

    svertices = vertices
    sfaces = faces
    sedges = edges
    shalfedges = halfedges

    def vertex_handle(self, index: int) -> VertexHandle:
        return VertexHandle(index)

    def edge_handle(self, value) -> EdgeHandle:
        if isinstance(value, HalfedgeHandle):
            self._ensure_topology()
            h = value.idx()
            return EdgeHandle(int(self._he_edge[h])) if 0 <= h < len(self._he_edge) else EdgeHandle(-1)
        return EdgeHandle(value)

    def face_handle(self, value) -> FaceHandle:
        if isinstance(value, HalfedgeHandle):
            self._ensure_topology()
            h = value.idx()
            return FaceHandle(int(self._he_face[h])) if 0 <= h < len(self._he_face) else FaceHandle(-1)
        return FaceHandle(value)

    def halfedge_handle(self, value, side=None) -> HalfedgeHandle:
        self._ensure_topology()
        if isinstance(value, EdgeHandle):
            return HalfedgeHandle(2 * value.idx() + int(side or 0))
        if isinstance(value, VertexHandle):
            v = value.idx()
            return HalfedgeHandle(int(self._vertex_out[v])) if 0 <= v < len(self._vertex_out) else HalfedgeHandle(-1)
        if isinstance(value, FaceHandle):
            f = value.idx()
            return HalfedgeHandle(int(self._face_he[f, 2])) if 0 <= f < len(self._face_he) else HalfedgeHandle(-1)
        return HalfedgeHandle(value)

    def is_valid_handle(self, handle) -> bool:
        limits = {
            VertexHandle: self.n_vertices,
            HalfedgeHandle: self.n_halfedges,
            EdgeHandle: self.n_edges,
            FaceHandle: self.n_faces,
        }
        return type(handle) in limits and 0 <= handle.idx() < limits[type(handle)]()

    def from_vertex_handle(self, hh: HalfedgeHandle) -> VertexHandle:
        self._ensure_topology()
        h = _idx(hh, HalfedgeHandle)
        return VertexHandle(int(self._he_from[h])) if 0 <= h < len(self._he_from) else VertexHandle(-1)

    def to_vertex_handle(self, hh: HalfedgeHandle) -> VertexHandle:
        self._ensure_topology()
        h = _idx(hh, HalfedgeHandle)
        return VertexHandle(int(self._he_to[h])) if 0 <= h < len(self._he_to) else VertexHandle(-1)

    def opposite_vh(self, hh: HalfedgeHandle) -> VertexHandle:
        return self.from_vertex_handle(hh)

    def opposite_halfedge_handle(self, hh: HalfedgeHandle) -> HalfedgeHandle:
        self._ensure_topology()
        h = _idx(hh, HalfedgeHandle)
        return HalfedgeHandle(int(self._he_opp[h])) if 0 <= h < len(self._he_opp) else HalfedgeHandle(-1)

    def next_halfedge_handle(self, hh: HalfedgeHandle) -> HalfedgeHandle:
        self._ensure_topology()
        h = _idx(hh, HalfedgeHandle)
        return HalfedgeHandle(int(self._he_next[h])) if 0 <= h < len(self._he_next) else HalfedgeHandle(-1)

    def prev_halfedge_handle(self, hh: HalfedgeHandle) -> HalfedgeHandle:
        self._ensure_topology()
        h = _idx(hh, HalfedgeHandle)
        return HalfedgeHandle(int(self._he_prev[h])) if 0 <= h < len(self._he_prev) else HalfedgeHandle(-1)

    def find_halfedge(self, from_vh: VertexHandle, to_vh: VertexHandle) -> HalfedgeHandle:
        target = to_vh.idx()
        for h in self._ring(from_vh):
            if self._he_to[h] == target:
                return HalfedgeHandle(int(h))
        return HalfedgeHandle(-1)

    def vv(self, vh: VertexHandle) -> Iterator[VertexHandle]:
        self._ensure_topology()
        return iter(VertexHandle(int(self._he_to[h])) for h in self._ring(vh))

    def voh(self, vh: VertexHandle) -> Iterator[HalfedgeHandle]:
        return iter(HalfedgeHandle(int(h)) for h in self._ring(vh))

    def vih(self, vh: VertexHandle) -> Iterator[HalfedgeHandle]:
        self._ensure_topology()
        return iter(HalfedgeHandle(int(self._he_opp[h])) for h in self._ring(vh))

    def ve(self, vh: VertexHandle) -> Iterator[EdgeHandle]:
        self._ensure_topology()
        return iter(EdgeHandle(int(self._he_edge[h])) for h in self._ring(vh))

    def vf(self, vh: VertexHandle) -> Iterator[FaceHandle]:
        self._ensure_topology()
        return iter(
            FaceHandle(int(self._he_face[h]))
            for h in self._ring(vh)
            if self._he_face[h] >= 0
        )

    def fv(self, fh: FaceHandle) -> Iterator[VertexHandle]:
        f = _idx(fh, FaceHandle)
        return iter(VertexHandle(int(v)) for v in self._faces[f])

    def fh(self, fh: FaceHandle) -> Iterator[HalfedgeHandle]:
        self._ensure_topology()
        f = _idx(fh, FaceHandle)
        hs = self._face_he[f]
        return iter(HalfedgeHandle(int(h)) for h in (hs[2], hs[0], hs[1]))

    def fe(self, fh: FaceHandle) -> Iterator[EdgeHandle]:
        self._ensure_topology()
        return iter(EdgeHandle(int(self._he_edge[h.idx()])) for h in self.fh(fh))

    def ff(self, fh: FaceHandle) -> Iterator[FaceHandle]:
        self._ensure_topology()
        result = []
        for h in self.fh(fh):
            other = self._he_face[self._he_opp[h.idx()]]
            if other >= 0:
                result.append(FaceHandle(int(other)))
        return iter(result)

    def is_boundary(self, handle, check_vertex=False) -> bool:
        self._ensure_topology()
        if isinstance(handle, HalfedgeHandle):
            return self._he_face[handle.idx()] < 0
        if isinstance(handle, EdgeHandle):
            return self._he_face[2 * handle.idx()] < 0 or self._he_face[2 * handle.idx() + 1] < 0
        if isinstance(handle, VertexHandle):
            h = self._vertex_out[handle.idx()]
            return h >= 0 and self._he_face[h] < 0
        if isinstance(handle, FaceHandle):
            if check_vertex:
                return any(self.is_boundary(v) for v in self.fv(handle))
            return any(self.is_boundary(self.edge_handle(h)) for h in self.fh(handle))
        raise TypeError("unsupported handle type")

    def valence(self, handle) -> int:
        if isinstance(handle, VertexHandle):
            return len(self._ring(handle))
        if isinstance(handle, FaceHandle):
            return 3
        raise TypeError("valence expects VertexHandle or FaceHandle")

    def points(self) -> np.ndarray:
        return self._points

    def point(self, vh: VertexHandle) -> np.ndarray:
        return self._points[_idx(vh, VertexHandle)]

    def set_point(self, vh: VertexHandle, point) -> None:
        self._points[_idx(vh, VertexHandle)] = point

    def face_vertex_indices(self) -> np.ndarray:
        return self._faces.astype(np.int32, copy=True)

    fv_indices = face_vertex_indices

    def edge_vertex_indices(self) -> np.ndarray:
        self._ensure_topology()
        return np.column_stack((self._he_from[::2], self._he_to[::2])).astype(np.int32)

    ev_indices = edge_vertex_indices

    def halfedge_vertex_indices(self) -> np.ndarray:
        self._ensure_topology()
        return np.column_stack((self._he_from, self._he_to)).astype(np.int32)

    hv_indices = halfedge_vertex_indices

    def halfedge_from_vertex_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._he_from.astype(np.int32)

    hfv_indices = halfedge_from_vertex_indices

    def halfedge_to_vertex_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._he_to.astype(np.int32)

    htv_indices = halfedge_to_vertex_indices

    def halfedge_face_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._he_face.astype(np.int32)

    hf_indices = halfedge_face_indices

    def halfedge_edge_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._he_edge.astype(np.int32)

    he_indices = halfedge_edge_indices

    def face_halfedge_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._face_he[:, [2, 0, 1]].astype(np.int32)

    fh_indices = face_halfedge_indices

    def face_edge_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._he_edge[self._face_he[:, [2, 0, 1]]].astype(np.int32)

    fe_indices = face_edge_indices

    def vertex_outgoing_halfedge_indices(self) -> np.ndarray:
        self._ensure_rings()
        degrees = np.diff(self._ring_offsets)
        width = int(degrees.max(initial=0))
        result = i64(shape=(self.n_vertices(), width))
        if width:
            lib().mom_padded_rows(
                self.n_vertices(),
                addr(self._ring_offsets),
                addr(self._ring_halfedges),
                width,
                addr(result),
            )
        return result.astype(np.int32)

    voh_indices = vertex_outgoing_halfedge_indices

    def vertex_incoming_halfedge_indices(self) -> np.ndarray:
        self._ensure_topology()
        return _dense([self._he_opp[self._ring(VertexHandle(v))] for v in range(self.n_vertices())])

    vih_indices = vertex_incoming_halfedge_indices

    def vertex_vertex_indices(self) -> np.ndarray:
        self._ensure_rings()
        degrees = np.diff(self._ring_offsets)
        width = int(degrees.max(initial=0))
        result = i64(shape=(self.n_vertices(), width))
        if width:
            lib().mom_padded_rows(
                self.n_vertices(),
                addr(self._ring_offsets),
                addr(self._ring_neighbors),
                width,
                addr(result),
            )
        return result.astype(np.int32)

    vv_indices = vertex_vertex_indices

    def vertex_edge_indices(self) -> np.ndarray:
        self._ensure_topology()
        return _dense([self._he_edge[self._ring(VertexHandle(v))] for v in range(self.n_vertices())])

    ve_indices = vertex_edge_indices

    def vertex_face_indices(self) -> np.ndarray:
        self._ensure_topology()
        rows = []
        for v in range(self.n_vertices()):
            rows.append(self._he_face[self._ring(VertexHandle(v))])
            rows[-1] = rows[-1][rows[-1] >= 0]
        return _dense(rows)

    vf_indices = vertex_face_indices

    def face_face_indices(self) -> np.ndarray:
        return _dense(
            [np.fromiter((h.idx() for h in self.ff(FaceHandle(f))), dtype=np.int64) for f in range(self.n_faces())]
        )

    ff_indices = face_face_indices

    def edge_halfedge_indices(self) -> np.ndarray:
        e = np.arange(self.n_edges(), dtype=np.int32)
        return np.column_stack((2 * e, 2 * e + 1))

    eh_indices = edge_halfedge_indices

    def edge_face_indices(self) -> np.ndarray:
        self._ensure_topology()
        return self._he_face.reshape(-1, 2).astype(np.int32)

    ef_indices = edge_face_indices

    def calc_edge_lengths(self) -> np.ndarray:
        self._ensure_topology()
        result = f64(shape=self.n_edges())
        if len(result):
            lib().mom_edge_lengths(
                addr(self._points), addr(self._he_from), addr(self._he_to), len(result), addr(result)
            )
        return result

    def calc_edge_length(self, eh: EdgeHandle) -> float:
        return float(self.calc_edge_lengths()[_idx(eh, EdgeHandle)])

    def calc_edge_sqr_length(self, eh: EdgeHandle) -> float:
        value = self.calc_edge_length(eh)
        return value * value

    def calc_edge_vector(self, eh: EdgeHandle) -> np.ndarray:
        a, b = self.edge_vertex_indices()[_idx(eh, EdgeHandle)]
        return self._points[b] - self._points[a]

    def calc_face_normals(self) -> np.ndarray:
        result = f64(shape=(self.n_faces(), 3))
        if len(result):
            lib().mom_face_normals(addr(self._points), addr(self._faces), len(result), addr(result))
        return result

    def calc_face_normal(self, fh: FaceHandle) -> np.ndarray:
        return self.calc_face_normals()[_idx(fh, FaceHandle)]

    def calc_face_centroid(self, fh: FaceHandle) -> np.ndarray:
        return self._points[self._faces[_idx(fh, FaceHandle)]].mean(axis=0)

    def _has_duplicate_faces(self, faces: np.ndarray) -> bool:
        canonical = {tuple(sorted(map(int, row))) for row in faces}
        return len(canonical) != len(faces)

    def is_flip_ok(self, eh: EdgeHandle) -> bool:
        self._ensure_topology()
        e = _idx(eh, EdgeHandle)
        if not 0 <= e < self.n_edges() or self.is_boundary(eh):
            return False
        h, o = 2 * e, 2 * e + 1
        c = int(self._he_to[self._he_next[h]])
        d = int(self._he_to[self._he_next[o]])
        if c == d:
            return False
        existing = self.find_halfedge(VertexHandle(c), VertexHandle(d))
        return not existing.is_valid()

    def flip(self, eh: EdgeHandle) -> None:
        self._ensure_topology()
        edge = _idx(eh, EdgeHandle)
        ok = self._flip_kernel(self._flip_address, self._edge_count, edge)
        if not ok:
            raise ValueError("edge cannot be flipped")
        self._rings_dirty = True

    def split(self, handle, vertex):
        if not isinstance(handle, EdgeHandle):
            raise NotImplementedError("covered split primitive expects an EdgeHandle")
        self._ensure_topology()
        if not self.is_valid_handle(handle):
            raise ValueError("invalid edge handle")
        created = not isinstance(vertex, VertexHandle)
        if created:
            vh = self.add_vertex(vertex)
            self._ensure_topology()
        else:
            vh = vertex
            if not self.is_valid_handle(vh):
                raise ValueError("invalid vertex handle")
        result = i64(shape=(self.n_faces() + 2, 3))
        count = lib().mom_split_edge(
            addr(self._faces),
            self.n_faces(),
            addr(self._he_from),
            addr(self._he_to),
            addr(self._he_next),
            addr(self._he_opp),
            addr(self._he_face),
            handle.idx(),
            vh.idx(),
            addr(result),
        )
        if not self.n_faces() <= count <= self.n_faces() + 2:
            raise RuntimeError("split kernel returned an invalid face count")
        try:
            self._replace_faces(result[:count])
        except Exception:
            if created:
                self._points = self._points[:-1].copy()
                self._invalidate()
            raise
        return vh if created else None

    split_edge = split
    split_edge_copy = split

    def _collapsed_candidate(self, hh: HalfedgeHandle):
        self._ensure_topology()
        h = _idx(hh, HalfedgeHandle)
        if not 0 <= h < self.n_halfedges():
            return None
        a, b = int(self._he_from[h]), int(self._he_to[h])
        if not self.is_boundary(self.edge_handle(hh)) and self.is_boundary(
            VertexHandle(a)
        ) and self.is_boundary(VertexHandle(b)):
            return None
        result = i64(shape=self._faces.shape)
        count = lib().mom_collapse_halfedge(
            addr(self._faces), self.n_faces(), self.n_vertices(), a, b, addr(result)
        )
        if not 0 <= count <= self.n_faces():
            raise RuntimeError("collapse kernel returned an invalid face count")
        faces = result[:count].copy()
        points = self._points.copy()
        if a != len(points) - 1:
            points[a] = points[-1]
        points = points[:-1].copy()
        if self._has_duplicate_faces(faces):
            return None
        candidate = TriMesh()
        candidate._points = np.ascontiguousarray(points)
        candidate._faces = np.ascontiguousarray(faces)
        candidate._invalidate()
        try:
            candidate._ensure_topology()
        except TopologyError:
            return None
        return points, faces

    def is_collapse_ok(self, hh: HalfedgeHandle) -> bool:
        return self._collapsed_candidate(hh) is not None

    def collapse(self, hh: HalfedgeHandle) -> None:
        candidate = self._collapsed_candidate(hh)
        if candidate is None:
            raise ValueError("halfedge cannot be collapsed")
        self._points, self._faces = candidate
        self._invalidate()
        self._ensure_topology()

    def garbage_collection(self, *args, **kwargs) -> None:
        pass

    def is_deleted(self, handle) -> bool:
        return False

    def request_vertex_status(self) -> None:
        pass

    def request_edge_status(self) -> None:
        pass

    def request_halfedge_status(self) -> None:
        pass

    def request_face_status(self) -> None:
        pass

    def is_triangles(self) -> bool:
        return True

    def is_trimesh(self) -> bool:
        return True

    def is_polymesh(self) -> bool:
        return False
