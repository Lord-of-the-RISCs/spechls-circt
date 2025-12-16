from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class PrintMuSymbolNamesPass(ModulePass):
    """Print `sym_name` for each `spechls.mu` op in the module."""

    name = "print-mu-symbol-names"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # Walk the whole IR rooted at the module (recursively).
        for nested_op in op.walk():
            if nested_op.name != "spechls.mu":
                continue

            sym = nested_op.attributes.get("sym_name")
            if isinstance(sym, builtin.StringAttr):
                print(sym.data)
            else:
                # Be robust to malformed/partial IR.
                print("<spechls.mu without StringAttr sym_name>")
