# Middle-end

import abc
import copy
import re
from typing import Self

from src.constants import Targets


class TensorReadWrite:
    def __init__(self, var: str, at: list[str], shape: list[str]):
        self.var = var
        self.at = at
        self.shape = shape


class Loop:
    def __init__(self, var: str, start: str, end: str):
        self.var = var
        self.start = start
        self.end = end


class MiddleEndOp(abc.ABC):
    def __init__(self):
        self.name: str

    def copy(self) -> Self:
        return copy.deepcopy(self)
    
    def __repr__(self) -> str:
        from src.backend import Reduction
        return Reduction().gen(self, target=Targets.IR)


class Reduction(MiddleEndOp):
    def __init__(
        self,
        name: str,
        loops: list[Loop],
        reads: list[TensorReadWrite],
        write: TensorReadWrite,
        reduction: str | None = None,
    ):
        self.name = name
        self.loops = loops
        self.reads: list[TensorReadWrite] = reads
        self.write: TensorReadWrite = write
        self.reduction = reduction

        # (loop_var, Read)
        self.promotions: list[tuple[str, TensorReadWrite]] = []
        self.vectorized = None
        self.unrolled = None

    def tile(self, tile_sizes: dict[str, int]) -> Self:
        def remap_indices(idxs: list[str], tile_map: dict) -> list[str]:
            new_idxs = []
            for idx in idxs:
                for sub, (blk, inner, T) in tile_map.items():
                    pattern = r'\b' + re.escape(sub) + r'\b'  # prohibiting recursive sub
                    idx = re.sub(pattern, f"({blk} * {T} + {inner})", idx)
                new_idxs.append(idx)
            return new_idxs

        new_loops = []
        tile_map = {}  # {old var: (blk, inner, tile_size)}
        for loop in self.loops:
            v = loop.var
            if v in tile_sizes:
                T = tile_sizes[v]
                blk = f"{v}_blk"
                inner = f"{v}_inner"
                new_loops.append(Loop(blk, str(loop.start), f"{loop.end}//{T}"))
                new_loops.append(Loop(inner, str(0), str(T)))
                tile_map[v] = (blk, inner, T)
            else:
                new_loops.append(loop)

        for r in self.reads:  # index remapping
            r.at = remap_indices(r.at, tile_map)
        self.write.at = remap_indices(self.write.at, tile_map)

        if self.reduction in tile_sizes:  # changing reduction symbol
            self.reduction = f"{self.reduction}_inner"

        self.loops = new_loops
        return self

    def reorder(self, new_order: list[str]) -> Self:
        loop_map = {loop.var: loop for loop in self.loops}
        self.loops = [loop_map[v] for v in new_order]
        return self

    def promote_invariants(self, loop_var: str) -> Self:
        invariant_reads: list[TensorReadWrite] = []
        for r in self.reads:
            if loop_var not in "".join(r.at):
                invariant_reads.append(r)
        if invariant_reads:
            for r in invariant_reads:
                self.promotions.append((loop_var, r))
            print(f"[Promotion] Hoisting {', '.join(r.var for r in invariant_reads)} outside {loop_var}")
        else:
            print(f"[Promotion] Nothing to hoist outside {loop_var}")
        return self

    def vectorize(self, loop, width) -> Self:
        ...
        return self

    def unroll(self, loop, factor) -> Self:
        ...
        return self
