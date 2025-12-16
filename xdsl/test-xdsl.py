# python
import sys
import os
import xdsl

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects.builtin import Builtin

from spechls_xdsl import SpecHLS
from PrintMuSymbolNames import PrintMuSymbolNamesPass
from xdsl.dialects import comb, seq, hw
#from hw_constant_ext import HWConstantOp
from hw_constant_ext import HW

ctx = Context()
ctx.allow_unregistered = True   # <-- add this
#ctx.register_op(HWConstantOp)
ctx.load_dialect(Builtin)
ctx.load_dialect(comb.Comb)  # comb dialect :contentReference[oaicite:0]{index=0}
ctx.load_dialect(seq.Seq)    # seq dialect :contentReference[oaicite:1]{index=1}
ctx.load_dialect(HW)      # hw dialect :contentReference[oaicite:2]{index=2}
ctx.load_dialect(SpecHLS)

if len(sys.argv) > 1:
    first_arg = sys.argv[1]
    print("first arg (argv[1]):", first_arg)
else:
    print("no args supplied (only arg0 present)")
module = Parser(ctx, open(first_arg).read()).parse_module()
print(module);
#PrintMuSymbolNamesPass().apply(ctx, module)
