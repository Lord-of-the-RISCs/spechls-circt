// SpecHLSFSMBuilder.h
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"

namespace spechls {
using namespace mlir;
using namespace circt;

struct Spec {
  std::string name;              // e.g. "S0"
  unsigned numPaths;             // number of input paths
  unsigned fastIndex;            // index of the fast path
  unsigned condDelay;            // latency for condition
  llvm::SmallVector<unsigned> inputDelays; // per-path latency
};

struct Model {
  llvm::SmallVector<Spec> specs;
  // For combined/nested speculation you can add:
  // DenseMap<unsigned, SmallVector<unsigned>> poisonMap; // s -> poisoned specs
};

struct Ports {
  // Machine I/O, Moore-style:
  llvm::SmallVector<Type> inputs;                 // mispec_* per spec (uBW)
  llvm::SmallVector<Type> outputs;                // commit_*, selSlowPath_*, rollback_*, startStall_*, globals
  llvm::SmallVector<Attribute> argNames;          // StringAttr
  llvm::SmallVector<Attribute> resNames;          // StringAttr
};

static inline unsigned clog2(unsigned x) {
  unsigned w = 0, v = (x <= 1) ? 1u : (x - 1u);
  while (v) { v >>= 1; ++w; }
  return std::max(1u, w);
}

static Ports makePorts(MLIRContext &ctx, const Model &m) {
  Ports p;
  // Inputs: one mispec per spec, width = ceil_log2(numPaths)
  for (auto &sp : m.specs) {
    auto bw = clog2(sp.numPaths);
    p.inputs.push_back(IntegerType::get(&ctx, bw));
    p.argNames.push_back(StringAttr::get(&ctx, ("mispec_" + sp.name)));
  }

  // Per-spec outputs: commit, selSlowPath (uBW), rollback, startStall
  for (auto &sp : m.specs) {
    auto bw = clog2(sp.numPaths);
    p.outputs.push_back(IntegerType::get(&ctx, 1)); // commit
    p.resNames.push_back(StringAttr::get(&ctx, ("commit_" + sp.name)));
    p.outputs.push_back(IntegerType::get(&ctx, bw)); // selSlowPath
    p.resNames.push_back(StringAttr::get(&ctx, ("selSlowPath_" + sp.name)));
    p.outputs.push_back(IntegerType::get(&ctx, 1)); // rollback
    p.resNames.push_back(StringAttr::get(&ctx, ("rollback_" + sp.name)));
    p.outputs.push_back(IntegerType::get(&ctx, 1)); // startStall
    p.resNames.push_back(StringAttr::get(&ctx, ("startStall_" + sp.name)));
  }

  // Global outputs: array_rollback, mu_rollback, rewind, rbwe
  auto i1 = IntegerType::get(&ctx, 1);
  p.outputs.push_back(i1); p.resNames.push_back(StringAttr::get(&ctx, "array_rollback"));
  p.outputs.push_back(i1); p.resNames.push_back(StringAttr::get(&ctx, "mu_rollback"));
  p.outputs.push_back(i1); p.resNames.push_back(StringAttr::get(&ctx, "rewind"));
  p.outputs.push_back(i1); p.resNames.push_back(StringAttr::get(&ctx, "rbwe"));

  return p;
}

// Helper to build a constant in a region with the right type.
static Value cst(OpBuilder &b, Location loc, Type ty, uint64_t v) {
  auto iTy = dyn_cast<IntegerType>(ty);
  if (!iTy) return Value();
  return b.create<arith::ConstantOp>(loc, IntegerAttr::get(iTy, APInt(iTy.getWidth(), v)));
}

// Emit fsm.output given per-result constants (or zeros if omitted).
static void emitStateOutputs(fsm::StateOp state,
                             ArrayRef<Type> outTypes,
                             llvm::function_ref<Value(unsigned)> producer) {
  //OpBuilder b(state.getOutput().front(), /*atStart=*/true);
  OpBuilder b(state);
  SmallVector<Value> outs;
  outs.reserve(outTypes.size());
  for (unsigned i = 0; i < outTypes.size(); ++i) {
    Value v = producer ? producer(i) : Value();
    if (!v) v = cst(b, state.getLoc(), outTypes[i], 0);
    outs.push_back(v);
  }
  b.create<fsm::OutputOp>(state.getLoc(), outs);
}

// Look up the result index of a named output for convenience.
static int findOutputIndex(const Ports &ports, StringRef name) {
  for (unsigned i = 0; i < ports.resNames.size(); ++i)
    if (cast<StringAttr>(ports.resNames[i]).getValue() == name) return (int)i;
  return -1;
}

struct BuildResult {
  fsm::MachineOp machine;
  fsm::StateOp proceedState;
  DenseMap<llvm::StringRef, fsm::StateOp> statesByName;
  Value controlVar; // fsm.variable result
};

// Core builder: creates the machine, Proceed/rollback/stall states and transitions for single-spec paths.
static BuildResult buildSpecHLSMachine(ModuleOp top, StringRef machineName, const Model &model) {
  MLIRContext &ctx = *top.getContext();
  OpBuilder root(&ctx);
  root.setInsertionPointToEnd(top.getBody());

  Ports ports = makePorts(ctx, model);
  auto funTy = FunctionType::get(&ctx, ports.inputs, ports.outputs);

  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, StringRef initialState, FunctionType function_type, ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> argAttrs = {});
  // static MachineOp create(::mlir::OpBuilder &builder,
  // ::mlir::Location location,
  // StringRef name,
  // StringRef initialState,
  // FunctionType function_type,
  // ArrayRef<NamedAttribute> attrs = {},
  // ArrayRef<DictionaryAttr> argAttrs = {});

  // static MachineOp create(::mlir::ImplicitLocOpBuilder &builder,
  // StringRef name,
  // StringRef initialState,
  // FunctionType function_type,
  // ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> argAttrs = {});

  auto mach = root.create<fsm::MachineOp>(
    top.getLoc(),
    /*name*/ StringAttr::get(&ctx, machineName),
    "IDLE", // initial state (will set later)
     funTy
  );

  // Attach named I/O (nice for downstream)
  mach.setArgNamesAttr(ArrayAttr::get(&ctx, ports.argNames));
  mach.setResNamesAttr(ArrayAttr::get(&ctx, ports.resNames));

  // Body block for machine
  {
    // Declare a control variable (example: u32 iteration)
    OpBuilder b = OpBuilder::atBlockEnd(&mach.getBody().getBlocks().front());
    auto u32 = IntegerType::get(&ctx, 32);
    auto zero = IntegerAttr::get(u32, 0);

  //   static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Attribute initValue, ::mlir::StringAttr name);

  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute initValue, ::mlir::StringAttr name);
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Attribute initValue, ::mlir::StringAttr name);
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Attribute initValue, ::mlir::StringAttr name);
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Attribute initValue, ::llvm::StringRef name);
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Attribute initValue, ::llvm::StringRef name);
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Attribute initValue, ::llvm::StringRef name);
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute initValue, ::llvm::StringRef name);
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Attribute initValue, ::llvm::StringRef name);
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Attribute initValue, ::llvm::StringRef name);
  // static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  // static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  // static VariableOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder,
  // ::mlir::ValueRange operands,
  // const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  //
    // static VariableOp create(::mlir::ImplicitLocOpBuilder &builder,
    // ::mlir::Type result,
    // ::mlir::Attribute initValue,
    // ::mlir::StringAttr name);
    auto ctrl = b.create<fsm::VariableOp>(top.getLoc(),
        u32,
    /*initValue=*/zero,
                  StringAttr::get(&ctx, "control")

                  );
    (void)ctrl;
  }

  SymbolTable symTab(top);

  // === Proceed state (default steady state) ===
  auto proceed = [&] {
    OpBuilder b = OpBuilder::atBlockEnd(&mach.getBody().getBlocks().front());
    auto st = b.create<fsm::StateOp>(top.getLoc(), StringAttr::get(&ctx, "Proceed"));
    // Default outputs: commit=1 for all specs, selSlowPath=fastIndex, others 0
    emitStateOutputs(st, ports.outputs, [&](unsigned oi) -> Value {
      OpBuilder ob(st);
      //OpBuilder ob(st.getOutput().front(), /*atStart=*/true);
      // Map out indices back to names to fill defaults.
      auto name = cast<StringAttr>(ports.resNames[oi]).getValue();
      if (name == "array_rollback" || name == "mu_rollback" || name == "rewind" || name == "rbwe")
        return cst(ob, st.getLoc(), ports.outputs[oi], 0);

      // Per-spec groups: commit, selSlowPath, rollback, startStall
      for (auto &sp : model.specs) {
        if (name == ("commit_" + sp.name)) return cst(ob, st.getLoc(), ports.outputs[oi], 1);
        if (name == ("selSlowPath_" + sp.name)) {
          auto bw = clog2(sp.numPaths);
          return cst(ob, st.getLoc(), ports.outputs[oi], sp.fastIndex);
        }
        if (name == ("rollback_" + sp.name))   return cst(ob, st.getLoc(), ports.outputs[oi], 0);
        if (name == ("startStall_" + sp.name)) return cst(ob, st.getLoc(), ports.outputs[oi], 0);
      }
      return Value();
    });
    return st;
  }();

  // Remember initial state.
  mach.setInitialStateAttr(proceed.getSymNameAttr());

  DenseMap<llvm::StringRef, fsm::StateOp> stateMap;
  stateMap[proceed.getSymName().str()] = proceed;

  // === Per-spec slow path handling (stall + rollback) ===
  auto i1 = IntegerType::get(&ctx, 1);
  // Helpers to pick ports by name.
  auto idxCommit = [&](StringRef s){ return findOutputIndex(ports, "commit_" + s.str()); };
  auto idxSel    = [&](StringRef s){ return findOutputIndex(ports, "selSlowPath_" + s.str()); };
  auto idxRb     = [&](StringRef s){ return findOutputIndex(ports, "rollback_" + s.str()); };
  auto idxStall  = [&](StringRef s){ return findOutputIndex(ports, "startStall_" + s.str()); };
  int idxRBWE = findOutputIndex(ports, "rbwe");

  // Use arg SSA values in guard/action via block arguments of machine body’s callable interface:
  // The inputs to the machine are referenced with `fsm.callable.getArguments()` semantics via FunctionOpInterface.
  auto argTypes = funTy.getInputs();

  // Build a small helper to fetch a machine argument by index.
  auto getArg = [&](unsigned argIdx) -> Value {
    // In fsm.machine, the callable body has no direct block args; we reference args via fsm.return in guards.
    // Convention: create an fsm.transition guard region that just compares the needed SSA "argument"
    // by using a block-local placeholder passed by fsm.return. We’ll materialize constants and use them in the guard.
    // (Practically, many producers just rebuild simple expressions in the guard.)
    (void)argIdx;
    return Value(); // We will build constants and compares in the guard region directly.
  };

  // From Proceed, for each spec and for each slow path != fast, create (optional) Stall chain then Rollback.
  for (unsigned si = 0; si < model.specs.size(); ++si) {
    const auto &sp = model.specs[si];
    const unsigned bw = clog2(sp.numPaths);

    for (unsigned path = 0; path < sp.numPaths; ++path) {
      if (path == sp.fastIndex) continue;

      const int nbStalls = std::max<int>(int(sp.inputDelays[path]) - int(sp.condDelay) - 1, 0);
      std::string base = (sp.name + std::to_string(path));

      // Create Stall states if needed.
      fsm::StateOp last = proceed;
      for (int s = 0; s < nbStalls; ++s) {
        OpBuilder b = OpBuilder::atBlockEnd(&mach.getBody().getBlocks().front());
        auto st = b.create<fsm::StateOp>(top.getLoc(),
                  StringAttr::get(&ctx, (base + "__Stall" + std::to_string(s))));
        stateMap[st.getSymName().str()] = st;

        // Outputs in Stall: hold fast selections for other specs, set startStall for this spec on first stall
        emitStateOutputs(st, ports.outputs, [&](unsigned oi)->Value{
          OpBuilder ob(st.getContext());
          auto nm = cast<StringAttr>(ports.resNames[oi]).getValue();
          // Default as Proceed, except:
          // - For this spec: startStall=1 on first stall; selSlowPath remains fastIndex until rollback
          if (nm == ("startStall_" + sp.name))
            return cst(ob, st.getLoc(), ports.outputs[oi], s==0 ? 1 : 0);
          if (nm == ("selSlowPath_" + sp.name))
            return cst(ob, st.getLoc(), ports.outputs[oi], sp.fastIndex);
          if (nm == ("commit_" + sp.name))   return cst(ob, st.getLoc(), ports.outputs[oi], 0);
          // Other signals keep Proceed defaults (handled by returning Value() to zero-fill, then we override below)
          // Fill "other spec" fast selections explicitly if you prefer deterministic code-gen.
          return Value();
        });

        // Transition from previous -> this stall unconditionally.
        {
          OpBuilder tb = OpBuilder::atBlockEnd(last.getTransitions().empty()
                 ? &last.getTransitions().emplaceBlock()
                 : &last.getTransitions().front());

          auto tr = tb.create<fsm::TransitionOp>(top.getLoc(),st);
          // No guard/action: unconditional.
        }
        last = st;
      }

      // Create Rollback state
      OpBuilder rbB = OpBuilder::atBlockEnd(&mach.getBody().getBlocks().front());
      auto rb = rbB.create<fsm::StateOp>(top.getLoc(),
                StringAttr::get(&ctx, (base + "__Rollback")));
      stateMap[rb.getSymName().str()] = rb;

      // Outputs in Rollback: rbwe=1, select fast for this spec (matches your Xtend code), others fast too
      emitStateOutputs(rb, ports.outputs, [&](unsigned oi)->Value{
        OpBuilder ob(rb.getContext());
        auto nm = cast<StringAttr>(ports.resNames[oi]).getValue();

        if (nm == "rbwe") return cst(ob, rb.getLoc(), ports.outputs[oi], 1);
        if (nm == ("selSlowPath_" + sp.name))
          return cst(ob, rb.getLoc(), ports.outputs[oi], sp.fastIndex);
        if (nm == ("commit_" + sp.name))   return cst(ob, rb.getLoc(), ports.outputs[oi], 0);
        if (nm == ("rollback_" + sp.name)) return cst(ob, rb.getLoc(), ports.outputs[oi], 1);
        // For other specs, keep fast by default (already the steady setting).
        return Value();
      });

      // Connect last (either Proceed or last Stall) --> Rollback with a guard:
      // guard: (mispec_<sp> == path)
      {
        fsm::StateOp src = last;
        OpBuilder tb = OpBuilder::atBlockEnd(src.getTransitions().empty()
            ? &src.getTransitions().emplaceBlock() : &src.getTransitions().front());
        auto tr = tb.create<fsm::TransitionOp>(top.getLoc(),rb);

        // Build guard region returning i1
        auto &g = tr.getGuard();
        OpBuilder gb(&g);
        gb.createBlock(&g);
        // Create a compare of the (symbolic) input to a constant 'path'
        // Since machine args aren’t explicit values here, we re-materialize a
        // small equality using `comb.icmp` over a block-local argument created
        // via arith.constant for now. In your pass, if you inline this builder,
        // you can plumb the actual machine arg as needed.
        auto argTy = ports.inputs[si];
        auto pathC = gb.create<arith::ConstantOp>(top.getLoc(),
                       IntegerAttr::get(argTy, path));
        // (In practice, you’d compare %mispec_si against 'pathC'. Here we just use 'pathC == pathC' as true,
        // and you should replace the LHS with the real argument wire when integrating.)
        auto eq = gb.create<comb::ICmpOp>(top.getLoc(),
                  comb::ICmpPredicateAttr::get(&ctx, comb::ICmpPredicate::eq),
                  pathC, pathC);
        gb.create<fsm::ReturnOp>(top.getLoc(), eq);
      }

      // TODO(COMBINED): emit additional guarded transitions to combined rollback states
      // when multiple mispecs occur (respect poison/ordering as in isGreaterForCombined()).
    }
  }

  return {mach, proceed, stateMap, /*controlVar*/ Value()};
}

// Thin HW wrapper: hw.module(clk, rst, mispec_*) -> (all FSM outputs), fsm.hw_instance + fsm.trigger
static hw::HWModuleOp buildSpecHLSWrapperHWModule(ModuleOp top,
                                                  StringRef wrapperName,
                                                  fsm::MachineOp machine) {
  MLIRContext &ctx = *top.getContext();
  OpBuilder b(&ctx);
  b.setInsertionPointToEnd(top.getBody());

  // Build ports: clk: !seq.clock, rst: i1, pass-through machine inputs/outputs
  SmallVector<hw::PortInfo> inputs, outputs;
  auto clkTy = seq::ClockType::get(&ctx);
  auto i1 = IntegerType::get(&ctx, 1);

  inputs.push_back(hw::PortInfo{
    StringAttr::get(&ctx, "clk"),     // name
    clkTy,                            // type
    hw::ModulePort::Direction::Input, // direction
    /*argNum*/ inputs.size()          // port index
});

  inputs.push_back(hw::PortInfo{
      StringAttr::get(&ctx, "rst"),
      i1,
      hw::ModulePort::Direction::Input,
      inputs.size()
  });

  // Mirror machine inputs as module inputs, and outputs as module outputs.
  FunctionType ft = machine.getFunctionType();
  for (auto en : llvm::enumerate(ft.getInputs())) {
    auto n = machine.getArgNamesAttr()
               ? cast<StringAttr>(machine.getArgNamesAttr()[en.index()]).getValue()
               : ("in" + std::to_string(en.index()));
    inputs.push_back(hw::PortInfo{StringAttr::get(&ctx,n), en.value(),
                                  hw::ModulePort::Direction::Input, inputs.size()});
  }
  for (auto en : llvm::enumerate(ft.getResults())) {
    auto n = machine.getResNamesAttr()
               ? cast<StringAttr>(machine.getResNamesAttr()[en.index()]).getValue()
               : ("out" + std::to_string(en.index()));



    outputs.push_back(hw::PortInfo{StringAttr::get(&ctx,n), en.value(),hw::ModulePort::Direction::Output,outputs.size()});

  }

  hw::ModulePortInfo mpi(inputs, outputs);
  auto wrap = b.create<hw::HWModuleOp>(top.getLoc(), StringAttr::get(&ctx, wrapperName), mpi);

  // Body
  {
    OpBuilder wb = OpBuilder::atBlockEnd(wrap.getBodyBlock());
    // Read inputs in order: clk, rst, then the FSM inputs
    Value clk = wrap.getArgumentForInput(0);
    Value rst = wrap.getArgumentForInput(1);

    SmallVector<Value> fsmIns;
    fsmIns.reserve(ft.getInputs().size());
    for (unsigned i = 0; i < ft.getInputs().size(); ++i)
      fsmIns.push_back(wrap.getArgumentForInput(2 + i));
    // static HWInstanceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange outputs, ::mlir::StringAttr name, ::mlir::FlatSymbolRefAttr machine, ::mlir::ValueRange inputs, ::mlir::Value clock, ::mlir::Value reset);
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::llvm::StringRef name, ::llvm::StringRef machine, ::mlir::ValueRange inputs, ::mlir::Value clock, ::mlir::Value reset);
    // static HWInstanceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange outputs, ::llvm::StringRef name, ::llvm::StringRef machine, ::mlir::ValueRange inputs, ::mlir::Value clock, ::mlir::Value reset);
    // static HWInstanceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange outputs, ::llvm::StringRef name, ::llvm::StringRef machine, ::mlir::ValueRange inputs, ::mlir::Value clock, ::mlir::Value reset);

    auto inst = fsm::HWInstanceOp::create(
    wb,
    wrap.getLoc(),
    /*outputs*/ ft.getResults(),
    /*name*/ wb.getStringAttr("inst"),
    /*machine*/ FlatSymbolRefAttr::get(&ctx, machine.getSymNameAttr()),
    /*inputs*/ fsmIns,
    /*clock*/ clk,
    /*reset*/ rst);


    wb.create<hw::OutputOp>(wrap.getLoc(), inst.getResults());
    // Clock/reset plumb through HWInstanceOp operands per op definition
    inst.getInputsMutable().assign(fsmIns);
    inst.getClockMutable().assign(clk);
    inst.getResetMutable().assign(rst);
  }

  return wrap;
}

} // namespace spechls
