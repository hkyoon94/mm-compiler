# BACK-END (CODEGEN)

import abc
from collections import defaultdict

import src.middleend as me
from src.constants import Commands, IR, Ops, Targets


class BackendOp(abc.ABC):
    def __init__(self):
        ...

    @abc.abstractmethod
    def gen(self, *args, **kwargs) -> str:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return self.gen(target=Targets.IR)


class Reduction(BackendOp):
    def __init__(self):
        ...

    def _emit(self, target: str, op: str, *v: str | me.TensorReadWrite | me.Loop) -> str:
        if target == Targets.IR:
            if op == Ops.LOAD:
                name, a = v
                return f"{name} = {IR.LOAD} {a.var}[{a.at[0]}, {a.at[1]}]"
            elif op == Ops.MUL:
                name, a, b = v
                return f"{name} = {IR.MUL} {a}, {b}"
            elif op == Ops.ACCUMUL:
                val, c = v
                return f"{IR.ACCUMUL} {val}, {c.var}[{c.at[0]}, {c.at[1]}]"
            elif op == Commands.FOR:
                loop, red_flag = v
                return f"{IR.FOR} ({loop.var} {IR.IN} {loop.start}..{loop.end}){red_flag} {{"
        
        elif target == Targets.C:
            if op == Ops.LOAD:
                name, a = v
                name = name.strip("%")
                return f"float {name} = {a.var}[({a.at[0]}) * ({a.shape[1]}) + {a.at[1]}];"
            elif op == Ops.MUL:
                name, a, b = v
                name, a, b = name.strip("%"), a.strip("%"), b.strip("%")
                return f"float {name} = {a} * {b};"
            elif op == Ops.ACCUMUL:
                val, c = v
                val = val.strip("%")
                return f"{c.var}[({c.at[0]}) * ({c.shape[1]}) + {c.at[1]}] += {val};"
            elif op == Commands.FOR:
                loop, red_flag = v
                return f"for (size_t {loop.var} = {eval(loop.start)}; {loop.var} < {eval(loop.end)}; ++{loop.var}) {{"

        else:
            raise ValueError(f"Unsupported target: {target}")

    def gen(self, op: me.Reduction, target: str) -> str:
        def new_ssa(var: str):
            ssa_counter[var] += 1
            return f"%{var.lower()}_{ssa_counter[var]}"

        if target == Targets.IR:
            code = f"@IR.{op.name}\n"
            indent = ""
        elif target == Targets.C:
            a, b = op.reads
            c = op.write
            code = f"// Compiled C++ code of legalized op: '{op.name}'\n"
            code += f"void {op.name}(const float* {a.var}, const float* {b.var}, float* {c.var}) {{\n"
            # code += f"size_t {a.shape[0]}, size_t {b.shape[1]}, size_t {a.shape[1]}) {{\n"
            indent = "  "
        symbol_table = {}
        ssa_counter = defaultdict(int) # var -> counter

        # building loops
        for loop in op.loops:
            for key, info in getattr(op, "index_hoists", {}).items():
                if info["scope"] == op.loops.index(loop):   # 해당 루프 스코프와 일치
                    code += f"{indent}const size_t {info['name']} = {info['expr']};\n"

            is_reduction = (loop.var == op.reduction)
            red_flag = " reduction" if is_reduction else ""

            # preload hoisted vars
            for (target_loop, r) in op.promotions:
                if target_loop == loop.var:
                    key = (r.var, tuple(r.at))
                    name = new_ssa(r.var)
                    symbol_table[key] = name
                    code += f"{indent}{self._emit(target, Ops.LOAD, name, r)}   // hoisted\n"

            code += f"{indent}{self._emit(target, Commands.FOR, loop, red_flag)}"
            if is_reduction:
                code += f"  // reduction over {loop.var}"
            code += "\n"
            indent += "  "

        w = op.write
        a, b = op.reads

        # SSA numbering init
        a_ssa_key = (a.var, tuple(a.at))
        b_ssa_key = (b.var, tuple(b.at))

        if a_ssa_key in symbol_table:
            a_name = symbol_table[a_ssa_key]
        else:
            a_name = new_ssa(a.var)
            code += f"{indent}{self._emit(target, Ops.LOAD, a_name, a)}\n"
            symbol_table[a_ssa_key] = a_name

        if b_ssa_key in symbol_table:
            b_name = symbol_table[b_ssa_key]
        else:
            b_name = new_ssa(b.var)
            code += f"{indent}{self._emit(target, Ops.LOAD, b_name, b)}\n"
            symbol_table[b_ssa_key] = b_name

        mul_name = new_ssa("mul")
        code += f"{indent}{self._emit(target, Ops.MUL, mul_name, a_name, b_name)}\n"

        # in-place accumulation (no explicit acc)
        code += f"{indent}{self._emit(target, Ops.ACCUMUL, mul_name, w)}\n"

        # loop close
        for _ in reversed(op.loops):
            indent = indent[:-2]
            code += f"{indent}}}\n"
        if target == Targets.C:
            code += "}\n"
        return code
