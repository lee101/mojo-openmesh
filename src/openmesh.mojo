"""Half-edge topology and triangle-mesh kernels exposed through a C ABI."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ip(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def fp(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def hash_slot(lo: Int64, hi: Int64, mask: Int) -> Int:
    return Int((lo * 1000003 ^ hi * 9176) & Int64(mask))


def fill_minus_one(pointer: I64Ptr, count: Int):
    comptime W = simd_width_of[DType.int64]()
    var packed = SIMD[DType.int64, W](-1)
    var i = 0
    var vector_stop = count - count % W
    while i < vector_stop:
        pointer.store(i, packed)
        i += W
    while i < count:
        pointer[i] = -1
        i += 1


@export("mom_build_topology")
def mom_build_topology(
    nv: Int,
    nf: Int,
    faces_addr: Int,
    hash_keys_addr: Int,
    hash_values_addr: Int,
    hash_capacity: Int,
    he_from_addr: Int,
    he_to_addr: Int,
    he_next_addr: Int,
    he_prev_addr: Int,
    he_opposite_addr: Int,
    he_face_addr: Int,
    he_edge_addr: Int,
    face_halfedges_addr: Int,
    vertex_out_addr: Int,
) abi("C") -> Int:
    var faces = ip(faces_addr)
    var hash_keys = ip(hash_keys_addr)
    var hash_values = ip(hash_values_addr)
    var he_from = ip(he_from_addr)
    var he_to = ip(he_to_addr)
    var he_next = ip(he_next_addr)
    var he_prev = ip(he_prev_addr)
    var he_opposite = ip(he_opposite_addr)
    var he_face = ip(he_face_addr)
    var he_edge = ip(he_edge_addr)
    var face_halfedges = ip(face_halfedges_addr)
    var vertex_out = ip(vertex_out_addr)

    fill_minus_one(hash_keys, hash_capacity)
    fill_minus_one(vertex_out, nv)

    var edge_count = 0
    var mask = hash_capacity - 1
    for f in range(nf):
        var v0 = Int(faces[3 * f])
        var v1 = Int(faces[3 * f + 1])
        var v2 = Int(faces[3 * f + 2])
        if (
            v0 < 0 or v0 >= nv or v1 < 0 or v1 >= nv or v2 < 0 or v2 >= nv
            or v0 == v1 or v1 == v2 or v2 == v0
        ):
            return -1

        for local in range(3):
            var a = v0 if local == 0 else (v1 if local == 1 else v2)
            var b = v1 if local == 0 else (v2 if local == 1 else v0)
            var lo = Int64(min(a, b))
            var hi = Int64(max(a, b))
            var key = lo * Int64(nv) + hi
            var slot = hash_slot(lo, hi, mask)
            var probes = 0
            while hash_keys[slot] != -1 and hash_keys[slot] != key:
                slot = (slot + 1) & mask
                probes += 1
                if probes >= hash_capacity:
                    return -2

            var h = 0
            if hash_keys[slot] == -1:
                var e = edge_count
                edge_count += 1
                h = 2 * e
                hash_keys[slot] = key
                hash_values[slot] = Int64(e)
                he_from[h] = Int64(a)
                he_to[h] = Int64(b)
                he_from[h + 1] = Int64(b)
                he_to[h + 1] = Int64(a)
                he_opposite[h] = Int64(h + 1)
                he_opposite[h + 1] = Int64(h)
                he_face[h] = -1
                he_face[h + 1] = -1
                he_edge[h] = Int64(e)
                he_edge[h + 1] = Int64(e)
                he_next[h] = -1
                he_next[h + 1] = -1
                he_prev[h] = -1
                he_prev[h + 1] = -1
            else:
                var e = Int(hash_values[slot])
                var h0 = 2 * e
                if Int(he_from[h0]) == a and Int(he_to[h0]) == b:
                    h = h0
                else:
                    h = h0 + 1
                if he_face[h] != -1:
                    return -3

            he_face[h] = Int64(f)
            face_halfedges[3 * f + local] = Int64(h)
            if vertex_out[a] == -1:
                vertex_out[a] = Int64(h)

        var h0 = Int(face_halfedges[3 * f])
        var h1 = Int(face_halfedges[3 * f + 1])
        var h2 = Int(face_halfedges[3 * f + 2])
        he_next[h0] = Int64(h1)
        he_next[h1] = Int64(h2)
        he_next[h2] = Int64(h0)
        he_prev[h0] = Int64(h2)
        he_prev[h1] = Int64(h0)
        he_prev[h2] = Int64(h1)

    var halfedge_count = 2 * edge_count
    for h in range(halfedge_count):
        if he_face[h] == -1:
            var a = Int(he_from[h])
            if vertex_out[a] != -1 and he_face[Int(vertex_out[a])] == -1:
                return -4
            vertex_out[a] = Int64(h)

    for h in range(halfedge_count):
        if he_face[h] == -1:
            var b = Int(he_to[h])
            var nh = Int(vertex_out[b])
            if nh < 0 or he_face[nh] != -1:
                return -4
            he_next[h] = Int64(nh)
            if he_prev[nh] != -1:
                return -4
            he_prev[nh] = Int64(h)

    return edge_count


@export("mom_vertex_rings")
def mom_vertex_rings(
    nv: Int,
    halfedge_count: Int,
    vertex_out_addr: Int,
    he_to_addr: Int,
    he_next_addr: Int,
    he_opposite_addr: Int,
    offsets_addr: Int,
    halfedges_addr: Int,
    neighbors_addr: Int,
) abi("C") -> Int:
    var vertex_out = ip(vertex_out_addr)
    var he_to = ip(he_to_addr)
    var he_next = ip(he_next_addr)
    var he_opposite = ip(he_opposite_addr)
    var offsets = ip(offsets_addr)
    var halfedges = ip(halfedges_addr)
    var neighbors = ip(neighbors_addr)
    var total = 0
    offsets[0] = 0
    for v in range(nv):
        var start = Int(vertex_out[v])
        if start >= 0:
            if start >= halfedge_count:
                return -1
            var h = start
            var degree = 0
            while True:
                if h < 0 or h >= halfedge_count or total >= halfedge_count:
                    return -1
                halfedges[total] = Int64(h)
                neighbors[total] = he_to[h]
                total += 1
                degree += 1
                var opposite = Int(he_opposite[h])
                if opposite < 0 or opposite >= halfedge_count:
                    return -1
                h = Int(he_next[opposite])
                if h == start:
                    break
                if degree > halfedge_count:
                    return -1
        offsets[v + 1] = Int64(total)
    return total


@export("mom_padded_rows")
def mom_padded_rows(
    row_count: Int,
    offsets_addr: Int,
    values_addr: Int,
    width: Int,
    result_addr: Int,
) abi("C"):
    var offsets = ip(offsets_addr)
    var values = ip(values_addr)
    var result = ip(result_addr)
    for row in range(row_count):
        var start = Int(offsets[row])
        var stop = Int(offsets[row + 1])
        var col = 0
        while start + col < stop:
            result[row * width + col] = values[start + col]
            col += 1
        while col < width:
            result[row * width + col] = -1
            col += 1


@export("mom_edge_lengths")
def mom_edge_lengths(
    points_addr: Int,
    he_from_addr: Int,
    he_to_addr: Int,
    edge_count: Int,
    lengths_addr: Int,
) abi("C"):
    var points = fp(points_addr)
    var he_from = ip(he_from_addr)
    var he_to = ip(he_to_addr)
    var lengths = fp(lengths_addr)
    for e in range(edge_count):
        var a = Int(he_from[2 * e])
        var b = Int(he_to[2 * e])
        var dx = points[3 * b] - points[3 * a]
        var dy = points[3 * b + 1] - points[3 * a + 1]
        var dz = points[3 * b + 2] - points[3 * a + 2]
        lengths[e] = sqrt(dx * dx + dy * dy + dz * dz)


@export("mom_face_normals")
def mom_face_normals(
    points_addr: Int,
    faces_addr: Int,
    face_count: Int,
    normals_addr: Int,
) abi("C"):
    var points = fp(points_addr)
    var faces = ip(faces_addr)
    var normals = fp(normals_addr)
    for f in range(face_count):
        var a = Int(faces[3 * f])
        var b = Int(faces[3 * f + 1])
        var c = Int(faces[3 * f + 2])
        var ux = points[3 * b] - points[3 * a]
        var uy = points[3 * b + 1] - points[3 * a + 1]
        var uz = points[3 * b + 2] - points[3 * a + 2]
        var vx = points[3 * c] - points[3 * a]
        var vy = points[3 * c + 1] - points[3 * a + 1]
        var vz = points[3 * c + 2] - points[3 * a + 2]
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        var norm = sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 0.0:
            nx /= norm
            ny /= norm
            nz /= norm
        normals[3 * f] = nx
        normals[3 * f + 1] = ny
        normals[3 * f + 2] = nz


@export("mom_flip_edge")
def mom_flip_edge(
    buffers_addr: Int,
    edge_count: Int,
    edge: Int,
) abi("C") -> Int:
    var buffers = ip(buffers_addr)
    var faces = ip(Int(buffers[0]))
    var he_from = ip(Int(buffers[1]))
    var he_to = ip(Int(buffers[2]))
    var he_next = ip(Int(buffers[3]))
    var he_prev = ip(Int(buffers[4]))
    var he_opposite = ip(Int(buffers[5]))
    var he_face = ip(Int(buffers[6]))
    var face_halfedges = ip(Int(buffers[7]))
    var vertex_out = ip(Int(buffers[8]))
    if edge < 0 or edge >= edge_count:
        return 0
    var h = 2 * edge
    if he_face[h] < 0:
        h += 1
    var o = Int(he_opposite[h])
    if he_face[h] < 0 or he_face[o] < 0:
        return 0
    var hn = Int(he_next[h])
    var hp = Int(he_prev[h])
    var on = Int(he_next[o])
    var op = Int(he_prev[o])
    var a = Int(he_from[h])
    var b = Int(he_to[h])
    var c = he_to[hn]
    var d = he_to[on]
    if c == d:
        return 0
    var start = Int(vertex_out[Int(c)])
    if start < 0:
        return 0
    var current = start
    while True:
        if he_to[current] == d:
            return 0
        current = Int(he_next[Int(he_opposite[current])])
        if current == start:
            break
        if current < 0:
            return 0
    var f0 = Int(he_face[h])
    var f1 = Int(he_face[o])
    faces[3 * f0] = c
    faces[3 * f0 + 1] = Int64(a)
    faces[3 * f0 + 2] = d
    faces[3 * f1] = d
    faces[3 * f1 + 1] = Int64(b)
    faces[3 * f1 + 2] = c

    he_from[h] = c
    he_to[h] = d
    he_from[o] = d
    he_to[o] = c

    he_next[hp] = Int64(on)
    he_next[on] = Int64(o)
    he_next[o] = Int64(hp)
    he_prev[hp] = Int64(o)
    he_prev[on] = Int64(hp)
    he_prev[o] = Int64(on)

    he_next[op] = Int64(hn)
    he_next[hn] = Int64(h)
    he_next[h] = Int64(op)
    he_prev[op] = Int64(h)
    he_prev[hn] = Int64(op)
    he_prev[h] = Int64(hn)

    he_face[hp] = Int64(f0)
    he_face[on] = Int64(f0)
    he_face[o] = Int64(f0)
    he_face[op] = Int64(f1)
    he_face[hn] = Int64(f1)
    he_face[h] = Int64(f1)
    face_halfedges[3 * f0] = Int64(hp)
    face_halfedges[3 * f0 + 1] = Int64(on)
    face_halfedges[3 * f0 + 2] = Int64(o)
    face_halfedges[3 * f1] = Int64(op)
    face_halfedges[3 * f1 + 1] = Int64(hn)
    face_halfedges[3 * f1 + 2] = Int64(h)

    if vertex_out[a] == Int64(h):
        vertex_out[a] = Int64(on)
    if vertex_out[b] == Int64(o):
        vertex_out[b] = Int64(hn)
    return 1


@export("mom_split_edge")
def mom_split_edge(
    faces_addr: Int,
    face_count: Int,
    he_from_addr: Int,
    he_to_addr: Int,
    he_next_addr: Int,
    he_opposite_addr: Int,
    he_face_addr: Int,
    edge: Int,
    new_vertex: Int,
    result_addr: Int,
) abi("C") -> Int:
    var faces = ip(faces_addr)
    var he_from = ip(he_from_addr)
    var he_to = ip(he_to_addr)
    var he_next = ip(he_next_addr)
    var he_opposite = ip(he_opposite_addr)
    var he_face = ip(he_face_addr)
    var result = ip(result_addr)
    for i in range(3 * face_count):
        result[i] = faces[i]
    var count = face_count
    var first = 2 * edge
    for side in range(2):
        var h = first if side == 0 else Int(he_opposite[first])
        var f = Int(he_face[h])
        if f >= 0:
            var a = he_from[h]
            var b = he_to[h]
            var c = he_to[Int(he_next[h])]
            result[3 * f] = a
            result[3 * f + 1] = Int64(new_vertex)
            result[3 * f + 2] = c
            result[3 * count] = Int64(new_vertex)
            result[3 * count + 1] = b
            result[3 * count + 2] = c
            count += 1
    return count


@export("mom_collapse_halfedge")
def mom_collapse_halfedge(
    faces_addr: Int,
    face_count: Int,
    vertex_count: Int,
    from_vertex: Int,
    to_vertex: Int,
    result_addr: Int,
) abi("C") -> Int:
    var faces = ip(faces_addr)
    var result = ip(result_addr)
    var count = 0
    var last_vertex = vertex_count - 1
    var target = from_vertex if to_vertex == last_vertex else to_vertex
    for f in range(face_count):
        var a = Int(faces[3 * f])
        var b = Int(faces[3 * f + 1])
        var c = Int(faces[3 * f + 2])
        if a == from_vertex:
            a = target
        elif a == last_vertex and from_vertex != last_vertex:
            a = from_vertex
        if b == from_vertex:
            b = target
        elif b == last_vertex and from_vertex != last_vertex:
            b = from_vertex
        if c == from_vertex:
            c = target
        elif c == last_vertex and from_vertex != last_vertex:
            c = from_vertex
        if a != b and b != c and c != a:
            result[3 * count] = Int64(a)
            result[3 * count + 1] = Int64(b)
            result[3 * count + 2] = Int64(c)
            count += 1
    return count
