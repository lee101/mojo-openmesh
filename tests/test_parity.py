from __future__ import annotations

import numpy as np
import pytest


POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def make(module, points=POINTS, faces=FACES):
    mesh = module.TriMesh()
    mesh.add_vertices(points)
    mesh.add_faces(faces)
    return mesh


def request_status(mesh):
    for name in (
        "request_vertex_status",
        "request_edge_status",
        "request_halfedge_status",
        "request_face_status",
    ):
        if hasattr(mesh, name):
            getattr(mesh, name)()


def grid(n):
    x, y = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(float)
    faces = []
    for row in range(n):
        for col in range(n):
            a = row * (n + 1) + col
            faces.extend(((a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)))
    return points, np.asarray(faces, dtype=np.int32)


def oriented_faces(mesh):
    result = []
    for a, b, c in mesh.face_vertex_indices():
        result.append(min((int(a), int(b), int(c)), (int(b), int(c), int(a)), (int(c), int(a), int(b))))
    return sorted(result)


def padded_row_sets(values):
    return [sorted(int(value) for value in row if value >= 0) for row in values]


def test_counts_match_upstream(om, upstream):
    ours, theirs = make(om), make(upstream)
    assert (ours.n_vertices(), ours.n_edges(), ours.n_faces(), ours.n_halfedges()) == (
        theirs.n_vertices(),
        theirs.n_edges(),
        theirs.n_faces(),
        theirs.n_halfedges(),
    )


@pytest.mark.parametrize(
    "method",
    [
        "face_vertex_indices",
        "edge_vertex_indices",
        "halfedge_vertex_indices",
        "halfedge_from_vertex_indices",
        "halfedge_to_vertex_indices",
        "face_halfedge_indices",
        "face_edge_indices",
        "halfedge_face_indices",
        "halfedge_edge_indices",
        "vertex_vertex_indices",
        "vertex_outgoing_halfedge_indices",
        "vertex_incoming_halfedge_indices",
        "vertex_face_indices",
        "vertex_edge_indices",
        "edge_halfedge_indices",
        "edge_face_indices",
        "face_face_indices",
    ],
)
def test_index_arrays_match_upstream(om, upstream, method):
    np.testing.assert_array_equal(getattr(make(om), method)(), getattr(make(upstream), method)())


@pytest.mark.parametrize("method", ["vv", "voh", "vih", "ve", "vf"])
def test_vertex_circulators_match_upstream(om, upstream, method):
    ours, theirs = make(om), make(upstream)
    for index in range(ours.n_vertices()):
        actual = [h.idx() for h in getattr(ours, method)(ours.vertex_handle(index))]
        expected = [h.idx() for h in getattr(theirs, method)(theirs.vertex_handle(index))]
        assert actual == expected


@pytest.mark.parametrize("method", ["fv", "fh", "fe", "ff"])
def test_face_circulators_match_upstream(om, upstream, method):
    ours, theirs = make(om), make(upstream)
    for index in range(ours.n_faces()):
        actual = [h.idx() for h in getattr(ours, method)(ours.face_handle(index))]
        expected = [h.idx() for h in getattr(theirs, method)(theirs.face_handle(index))]
        assert actual == expected


def test_halfedge_navigation_matches_upstream(om, upstream):
    ours, theirs = make(om), make(upstream)
    methods = (
        "from_vertex_handle",
        "to_vertex_handle",
        "opposite_halfedge_handle",
        "next_halfedge_handle",
        "prev_halfedge_handle",
        "face_handle",
        "edge_handle",
    )
    for index in range(ours.n_halfedges()):
        for method in methods:
            assert getattr(ours, method)(ours.halfedge_handle(index)).idx() == getattr(
                theirs, method
            )(theirs.halfedge_handle(index)).idx()


def test_boundary_flags_match_upstream(om, upstream):
    ours, theirs = make(om), make(upstream)
    for handle_name, count_name in (
        ("vertex_handle", "n_vertices"),
        ("edge_handle", "n_edges"),
        ("halfedge_handle", "n_halfedges"),
        ("face_handle", "n_faces"),
    ):
        for index in range(getattr(ours, count_name)()):
            assert ours.is_boundary(getattr(ours, handle_name)(index)) == theirs.is_boundary(
                getattr(theirs, handle_name)(index)
            )


def test_geometry_kernels_match_upstream(om, upstream):
    ours, theirs = make(om), make(upstream)
    np.testing.assert_allclose(
        ours.calc_edge_lengths(),
        [theirs.calc_edge_length(eh) for eh in theirs.edges()],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        ours.calc_face_normals(),
        [theirs.calc_face_normal(fh) for fh in theirs.faces()],
        rtol=0,
        atol=1e-14,
    )
    for index in range(ours.n_edges()):
        np.testing.assert_allclose(
            ours.calc_edge_vector(ours.edge_handle(index)),
            theirs.calc_edge_vector(theirs.edge_handle(index)),
        )
    for index in range(ours.n_faces()):
        expected = theirs.points()[theirs.face_vertex_indices()[index]].mean(axis=0)
        np.testing.assert_allclose(
            ours.calc_face_centroid(ours.face_handle(index)), expected
        )


def test_rejects_lossy_face_index_conversions(om):
    mesh = om.TriMesh()
    mesh.add_vertices(np.zeros((3, 3)))
    with pytest.raises(TypeError, match="integers"):
        mesh.add_faces(np.array([[0.0, 1.0, 2.5]]))
    with pytest.raises(OverflowError, match="int64"):
        mesh.add_faces(np.array([[0, 1, 2**63]], dtype=np.uint64))
    with pytest.raises(TypeError, match="integer"):
        mesh.add_face(0, 1, 1.5)
    assert mesh.n_faces() == 0


def test_ffi_buffer_contract():
    from mojo_openmesh._lib import addr

    with pytest.raises(ValueError, match="non-empty"):
        addr(np.empty(0, dtype=np.int64))
    with pytest.raises(TypeError, match="dtype"):
        addr(np.ones(1, dtype=np.int32))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((2, 2), dtype=np.float64)[:, ::2])
    readonly = np.ones(1, dtype=np.float64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        addr(readonly)


def test_split_rejects_invalid_vertex_without_calling_kernel(om):
    mesh = make(om)
    edge = mesh.edge_handle(
        mesh.find_halfedge(mesh.vertex_handle(0), mesh.vertex_handle(2))
    )
    with pytest.raises(ValueError, match="invalid vertex"):
        mesh.split(edge, om.VertexHandle(99))
    assert mesh.n_vertices() == 4
    assert mesh.n_faces() == 2


def test_grid_topology_matches_upstream(om, upstream):
    points, faces = grid(18)
    ours, theirs = make(om, points, faces), make(upstream, points, faces)
    for method in (
        "edge_vertex_indices",
        "face_halfedge_indices",
        "halfedge_face_indices",
    ):
        np.testing.assert_array_equal(getattr(ours, method)(), getattr(theirs, method)())
    assert padded_row_sets(ours.vertex_vertex_indices()) == padded_row_sets(
        theirs.vertex_vertex_indices()
    )
    assert padded_row_sets(ours.vertex_face_indices()) == padded_row_sets(
        theirs.vertex_face_indices()
    )


def test_topology_simd_tail_handles_isolated_vertices(om):
    mesh = om.TriMesh()
    points = np.zeros((11, 3), dtype=np.float64)
    points[1, 0] = 1.0
    points[2, 1] = 1.0
    mesh.add_vertices(points)
    mesh.add_faces(np.array([[0, 1, 2]], dtype=np.int64))
    assert mesh.n_edges() == 3
    rings = mesh.vertex_vertex_indices()
    assert rings.shape == (11, 2)
    assert np.all(rings[3:] == -1)


def test_large_topology_matches_upstream(om, upstream):
    points, faces = grid(128)
    ours, theirs = make(om, points, faces), make(upstream, points, faces)
    assert (ours.n_edges(), ours.n_halfedges()) == (
        theirs.n_edges(),
        theirs.n_halfedges(),
    )
    np.testing.assert_array_equal(
        ours.edge_vertex_indices(), theirs.edge_vertex_indices()
    )


def test_flip_matches_upstream(om, upstream):
    ours, theirs = make(om), make(upstream)
    for mesh in (ours, theirs):
        edge = mesh.edge_handle(mesh.find_halfedge(mesh.vertex_handle(0), mesh.vertex_handle(2)))
        assert mesh.is_flip_ok(edge)
        mesh.flip(edge)
    assert oriented_faces(ours) == oriented_faces(theirs)
    assert {tuple(sorted(edge)) for edge in ours.edge_vertex_indices()} == {
        tuple(sorted(edge)) for edge in theirs.edge_vertex_indices()
    }


def test_flip_updates_connectivity_and_rebuilds_rings_lazily(om, upstream):
    points, faces = grid(20)
    ours, theirs = make(om, points, faces), make(upstream, points, faces)
    center = 10 * 21 + 10
    diagonal = center + 22
    ours_edge = ours.edge_handle(
        ours.find_halfedge(ours.vertex_handle(center), ours.vertex_handle(diagonal))
    )
    theirs_edge = theirs.edge_handle(
        theirs.find_halfedge(
            theirs.vertex_handle(center), theirs.vertex_handle(diagonal)
        )
    )
    halfedges_id = id(ours._he_from)
    ring_offsets_id = id(ours._ring_offsets)
    ours.flip(ours_edge)
    theirs.flip(theirs_edge)
    assert id(ours._he_from) == halfedges_id
    assert id(ours._ring_offsets) == ring_offsets_id
    assert oriented_faces(ours) == oriented_faces(theirs)
    assert padded_row_sets(ours.vertex_vertex_indices()) == padded_row_sets(
        theirs.vertex_vertex_indices()
    )
    assert id(ours._ring_offsets) != ring_offsets_id


def test_flip_rejects_boundary_edge_without_mutation(om):
    mesh = make(om)
    edge = mesh.edge_handle(
        mesh.find_halfedge(mesh.vertex_handle(0), mesh.vertex_handle(1))
    )
    faces = mesh.face_vertex_indices()
    assert not mesh.is_flip_ok(edge)
    with pytest.raises(ValueError, match="cannot be flipped"):
        mesh.flip(edge)
    np.testing.assert_array_equal(mesh.face_vertex_indices(), faces)


@pytest.mark.parametrize(
    ("vertices", "point"),
    [((0, 2), [0.5, 0.5, 0.0]), ((0, 1), [0.5, 0.0, 0.0])],
)
def test_edge_split_matches_upstream(om, upstream, vertices, point):
    ours, theirs = make(om), make(upstream)
    for mesh in (ours, theirs):
        new_vertex = mesh.add_vertex(np.asarray(point))
        edge = mesh.edge_handle(
            mesh.find_halfedge(mesh.vertex_handle(vertices[0]), mesh.vertex_handle(vertices[1]))
        )
        assert mesh.split(edge, new_vertex) is None
    assert oriented_faces(ours) == oriented_faces(theirs)
    assert {tuple(sorted(edge)) for edge in ours.edge_vertex_indices()} == {
        tuple(sorted(edge)) for edge in theirs.edge_vertex_indices()
    }
    np.testing.assert_allclose(ours.points(), theirs.points())


@pytest.mark.parametrize("vertices", [(0, 1), (2, 3)])
def test_collapse_matches_upstream(om, upstream, vertices):
    ours, theirs = make(om), make(upstream)
    for mesh in (ours, theirs):
        request_status(mesh)
        halfedge = mesh.find_halfedge(
            mesh.vertex_handle(vertices[0]), mesh.vertex_handle(vertices[1])
        )
        assert mesh.is_collapse_ok(halfedge)
        mesh.collapse(halfedge)
        mesh.garbage_collection()
    assert oriented_faces(ours) == oriented_faces(theirs)
    assert {tuple(sorted(edge)) for edge in ours.edge_vertex_indices()} == {
        tuple(sorted(edge)) for edge in theirs.edge_vertex_indices()
    }
    np.testing.assert_allclose(ours.points(), theirs.points())


def test_rejects_duplicate_or_nonmanifold_faces(om):
    mesh = make(om)
    assert not mesh.add_face(mesh.vertex_handle(0), mesh.vertex_handle(1), mesh.vertex_handle(2)).is_valid()
    assert mesh.n_faces() == 2


def test_handle_contract(om):
    handle = om.VertexHandle(7)
    assert handle.idx() == 7 and handle.is_valid() and int(handle) == 7
    handle.invalidate()
    assert handle.idx() == -1 and not handle.is_valid()
    assert om.EdgeHandle(2) == om.EdgeHandle(2)
    assert om.EdgeHandle(2) != om.FaceHandle(2)


def test_empty_and_isolated_vertices(om):
    mesh = om.TriMesh()
    mesh.add_vertices(np.zeros((3, 3)))
    assert mesh.n_edges() == mesh.n_faces() == mesh.n_halfedges() == 0
    assert mesh.vertex_vertex_indices().shape == (3, 0)
    assert all(list(mesh.vv(vh)) == [] for vh in mesh.vertices())


def test_points_and_point_updates_match_upstream(om, upstream):
    ours, theirs = make(om), make(upstream)
    point = np.array([2.0, 3.0, 4.0])
    ours.set_point(ours.vertex_handle(1), point)
    theirs.set_point(theirs.vertex_handle(1), point)
    np.testing.assert_array_equal(ours.point(ours.vertex_handle(1)), point)
    np.testing.assert_array_equal(ours.points(), theirs.points())


def test_faces_without_vertices_fail_cleanly(om):
    mesh = om.TriMesh()
    with pytest.raises(om.TopologyError, match="invalid"):
        mesh.add_faces(np.array([[0, 1, 2]], dtype=np.int64))


def test_find_halfedge_and_valence(om, upstream):
    ours, theirs = make(om), make(upstream)
    for a in range(4):
        assert ours.valence(ours.vertex_handle(a)) == theirs.valence(theirs.vertex_handle(a))
        for b in range(4):
            assert ours.find_halfedge(ours.vertex_handle(a), ours.vertex_handle(b)).idx() == (
                theirs.find_halfedge(theirs.vertex_handle(a), theirs.vertex_handle(b)).idx()
            )
