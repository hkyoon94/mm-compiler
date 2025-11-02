import traceback

import src.backend as be
import src.middleend as me
from src.constants import Targets


REDUCTION_OP_WRAPPER = r"""
torch::Tensor c_reduction_wrapper_<op_tag>(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dimension mismatch");

  const auto M = A.size(0);
  const auto N = B.size(1);
  auto C = torch::zeros({M, N}, torch::kFloat);

  <op_tag>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}
"""

C_BINDING = r"""
PYBIND11_MODULE(<mod_tag>, m) {
"""


class CompiledSource:
    def __init__(self, name: str, source: str, backend: Targets):
        self.name = name
        self.source = source
        self.backend = backend

    def __repr__(self) -> str:
        return self.source


def compile(op: me.MiddleEndOp, target: Targets, name: str = None) -> str:
    if isinstance(op, me.Reduction):
        codegen = be.Reduction()
    else:
        raise ValueError(f"Unsupported middle-end op '{op.name}'")
    name = name if name is not None else op.name
    code = codegen.gen(op, target)
    code = code.replace(op.name, name)
    return CompiledSource(name, code, target)


class Runtime:
    KNOWN_BACKENDS = (Targets.C,)

    @staticmethod
    def _compile_c(sources: list[CompiledSource]) -> tuple[object, str]:
        from torch.utils.cpp_extension import load
        code = "// This code is auto-generated.\n"
        code += "#include <torch/extension.h>\n\n\n"

        for source in sources:
            code += source.source
            code += REDUCTION_OP_WRAPPER.replace("<op_tag>", source.name) + "\n"

        code += C_BINDING.replace("<mod_tag>", "mod")
        for source in sources:
            code += f'  m.def("{source.name}", &c_reduction_wrapper_{source.name}, "");\n'
        code += "}\n"

        fp = "./output/mod.cpp"
        with open(fp, "w") as f:
            f.write(code)

        try:  # JIT compile inline
            module = load(
                name="mod",
                sources=[fp],
                extra_cflags=[
                    "-O3",
                    "-march=native",
                    "-flto",
                ],
                extra_ldflags=[
                    "-flto",
                ],
                verbose=True,
            )
        except Exception:
            print(traceback.format_exc())
            module = None
        return module, code

    @classmethod
    def jit(cls, sources: list[CompiledSource] | CompiledSource) -> tuple[object, str]:
        if not isinstance(sources, list):
            sources = [sources]

        be = sources[0].backend
        names = []
        for s in sources:
            if s.backend != be:
                raise ValueError("Target backends for sources must be all same")
            names.append(s.name)
        if be not in cls.KNOWN_BACKENDS:
            raise ValueError(f"Unknown target backend '{be}'")

        if len(set(names)) != len(sources):
            raise ValueError("Each compiled source must have distinct name")

        if be == Targets.C:
            return cls._compile_c(sources)
