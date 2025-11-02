import abc
import re

import src.middleend as me


class FrontEndOp(abc.ABC):
    def __init__(self, ir: str):
        self.ir = ir

    @abc.abstractmethod
    def legalize(self) -> me.MiddleEndOp:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return self.ir + "\n"


class Reduction(FrontEndOp):
    def __init__(self, ir: str):
        self.ir = ir

    def legalize(self) -> me.Reduction:
        ir = self.ir.strip()
        m = re.match(r"\[(\d+),\s*(\d+)\]\s*@\s*\[(\d+),\s*(\d+)\]", ir)
        
        M, K1, K2, N = map(int, m.groups())
        if K1 != K2:
            raise ValueError(f"Incompatible shapes: inner dims {K1} and {K2} do not match")

        M, K1, N = str(M), str(K1), str(N)
        # legalize loops
        loops = [
            me.Loop(var="m", start=str(0), end=M),
            me.Loop(var="n", start=str(0), end=N),
            me.Loop(var="k", start=str(0), end=K1),
        ]
        # legalize Tensors
        A = me.TensorReadWrite(var="A", at=["m", "k"], shape=[M, K1])
        B = me.TensorReadWrite(var="B", at=["k", "n"], shape=[K1, N])
        C = me.TensorReadWrite(var="C", at=["m", "n"], shape=[M, N])
        
        return me.Reduction(
            name="mm",
            loops=loops,
            reads=[A, B],
            write=C,
            reduction="k",
        )


def parse(ir: str) -> FrontEndOp:
    """
    Very lightweight parser that examines the IR string and returns
    an appropriate FrontEndOp (currently only supports Reduction).
    """
    ir = ir.strip()

    # Simple dispatch: detect pattern "[M, K] @ [K, N]"
    if re.match(r"\[\s*\d+\s*,\s*\d+\s*\]\s*@\s*\[\s*\d+\s*,\s*\d+\s*\]", ir):
        return Reduction(ir)

    # (Optional) future: detect add, relu, or convolution patterns
    # elif re.match(r"conv2d", ir): ...
    # elif re.match(r"\[\s*\d+\s*,\s*\d+\]\s*\+\s*\[\s*\d+\s*,\s*\d+\]", ir): ...
    #     return ElementwiseAdd(ir)

    raise ValueError(f"Unknown or unsupported IR pattern: {ir}")
