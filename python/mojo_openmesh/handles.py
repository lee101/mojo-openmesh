from __future__ import annotations


class BaseHandle:
    __slots__ = ("_idx",)

    def __init__(self, idx: int = -1):
        self._idx = int(idx)

    def idx(self) -> int:
        return self._idx

    def is_valid(self) -> bool:
        return self._idx >= 0

    def invalidate(self) -> None:
        self._idx = -1

    def reset(self) -> None:
        self._idx = -1

    def __index__(self) -> int:
        return self._idx

    def __int__(self) -> int:
        return self._idx

    def __hash__(self) -> int:
        return hash((type(self), self._idx))

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._idx == other._idx

    def __lt__(self, other) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._idx < other._idx

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self._idx})>"


class VertexHandle(BaseHandle):
    pass


class HalfedgeHandle(BaseHandle):
    pass


class EdgeHandle(BaseHandle):
    pass


class FaceHandle(BaseHandle):
    pass
