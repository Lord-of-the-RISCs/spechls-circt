//===- ExportWcetCpp.cpp - wcet.core to C++ Core analysis -----------------===//
//
// Translate a wcet.core operation into a self-contained c++  file that
// implements the CoreAnalysis interface (compilewcet.h)
//
// Registration:
//  spechls-translate --export-wcet-cpp foo.mlir -o foo.cpp
//
//===----------------------------------------------------------------------===//

#include "Target/ExportWcetCpp/ExportWcetCpp.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace circt;

namespace {

// ----------------------------------------------------------------------------
// Helpers : MLIR type -> c++ unknown_ap_* type string
// ----------------------------------------------------------------------------

/// Returns the C++ unknown type string for a given MLIR type.
/// e.g. i32                    -> "abstract_ap_int<32>"
///      i1                     -> "abstract_bool"
///      !spechls.array<i32, 4> -> "abstract_ap_int_tab<32>"
static std::string mlirTypeToCpp(Type t) {
  if (auto itype = dyn_cast_or_null<IntegerType>(t)) {
    if (itype.getWidth() == 1)
      return "abstract_bool";
    if (itype.isSigned())
      return "abstract_ap_int<" + std::to_string(itype.getWidth()) + ">";
    return "abstract_ap_uint<" + std::to_string(itype.getWidth()) + ">";
  }
  if (auto arr = dyn_cast_or_null<spechls::ArrayType>(t)) {
    auto elem = cast<IntegerType>(arr.getElementType());
    if (elem.isSigned())
      return "abstract_ap_int_tab<" + std::to_string(elem.getWidth()) + ">";
    return "abstract_ap_uint_tab<" + std::to_string(elem.getWidth()) + ">";
  }
  return "/* unknown type */";
}

// Returns the "reset" expression for a type when a pipeline flush occurs:
//   abstract_ap_uint<W>  → abstract_ap_uint<W>()      (unknown)
//   abstract_ap_uint_tab → abstract_ap_uint_tab<W>(N)  (unknown)
//   abstract_bool        → abstract_bool(false)         (false, not unknown)
static std::string mlirTypeReset(Type t) {
  if (auto itype = dyn_cast<IntegerType>(t)) {
    if (itype.getWidth() == 1)
      return "abstract_bool()";
    return "abstract_ap_uint<" + std::to_string(itype.getWidth()) + ">()";
  }
  if (auto arr = dyn_cast<spechls::ArrayType>(t)) {
    auto elem = cast<IntegerType>(arr.getElementType());
    return "abstract_ap_uint_tab<" + std::to_string(elem.getWidth()) + ">(" + std::to_string(arr.getSize()) + ")";
  }
  return "/* unknown reset */";
}

// ----------------------------------------------------------------------------
// WcetCppEmitter
// ----------------------------------------------------------------------------

class WcetCppEmitter {
public:
  explicit WcetCppEmitter(raw_ostream &os, spechls::WcetTranslateOptions options) : os(os) {
    if (options.selectVersions == 1) {
      version = Version::V1;
    } else {
      version = Version::V2;
    }
  }

  LogicalResult translate(ModuleOp module);

private:
  enum Version { V1, V2 };

  Version version;
  raw_ostream &os;

  // SSA value -> C++ variable name
  llvm::DenseMap<Value, std::string> valueNames;
  unsigned varCounter = 0;

  // Information extracted from wcet.core signature
  struct ArgInfo {
    std::string name;     // C++ field name  (e.g. "delay2")
    std::string cppType;  // e.g. "abstract_ap_uint<2>"
    Type mlirType;        // original MLIR type (needed for reset exprs)
    bool isInstr = false; // has wcet.instrNb attribute
    bool isDelay = false; // has wcet.nbPred attribute
    int nbPred = -1;      // value of wcet.nbPred  (pipeline depth)
    // Position index among delay args (0-based, in declaration order)
    int delayPos = -1;
  };

  std::vector<ArgInfo> inArgs;  // all non-state input args
  std::vector<ArgInfo> outArgs; // all output args

  // FSM Information from spechls.fsm
  struct FsmInfo {
    std::vector<std::string> gammaNames;          // in FSM declaration order
    std::vector<std::vector<unsigned>> penTables; // per-gamma penalty vectors
    std::vector<std::string> packFieldNames;      // _inFSM field names
    std::vector<std::string> packFieldTypes;      // _inFSM field C++ types
    std::vector<Value> packOperandValues;         // SSA values packed into _inFSM
  };

  FsmInfo fsmInfo;

  // SSA values from wcet.commit, in order (map to outArgs fields)
  std::vector<Value> commitOperands;

  // Translation helpers
  std::string nameOf(Value v);
  std::string newVar(StringRef prefix = "v");

  // Per-op emitters
  LogicalResult translateWcetCore(wcet::CoreOp coreOp);

  // ---- Section emitters ----
  void emitFileHeader(StringRef className);
  void emitInStateStruct();
  void emitOutStateStruct();
  void emitInFsmStruct();
  void emitSetupAnalysis();
  void emitSetupNextState(wcet::CommitOp commitOp);
  void emitIsEqual();
  void emitClearInState();
  void emitClearOutState();
  void emitPrintOutState();
  void emitFsmFunction();
  void emitCoreFunction(wcet::CoreOp coreOp);
  void emitCoreNextFunction(wcet::CoreOp coreOp);
  void emitClassFooter(StringRef className);

  // ---- Per-operation emitters (used inside _core) ----
  LogicalResult emitOp(Operation *op);
  LogicalResult emitConstant(hw::ConstantOp op);
  LogicalResult emitHwBitcast(hw::BitcastOp op);
  LogicalResult emitCombConcat(comb::ConcatOp op);
  LogicalResult emitCombExtract(comb::ExtractOp op);
  LogicalResult emitCombICmp(comb::ICmpOp op);
  LogicalResult emitCombAnd(comb::AndOp op);
  LogicalResult emitCombOr(comb::OrOp op);
  LogicalResult emitCombMux(comb::MuxOp op);
  LogicalResult emitCombAdd(comb::AddOp op);
  LogicalResult emitCombMul(comb::MulOp op);
  LogicalResult emitCombSub(comb::SubOp op);
  LogicalResult emitCombDivU(comb::DivUOp op);
  LogicalResult emitCombDivS(comb::DivSOp op);
  LogicalResult emitCombModU(comb::ModUOp op);
  LogicalResult emitCombModS(comb::ModSOp op);
  LogicalResult emitCombXor(comb::XorOp op);
  LogicalResult emitCombShl(comb::ShlOp op);
  LogicalResult emitCombShrU(comb::ShrUOp op);
  LogicalResult emitCombShrS(comb::ShrSOp op);
  LogicalResult emitCombParity(comb::ParityOp op);
  LogicalResult emitCombReplicate(comb::ReplicateOp op);
  LogicalResult emitCombReverse(comb::ReverseOp op);
  // LogicalResult emitCombTruthTable(comb::TruthTableOp op);
  LogicalResult emitSpechlsLoad(spechls::LoadOp op);
  LogicalResult emitSpechlsAlpha(spechls::AlphaOp op);
  LogicalResult emitWcetGamma(wcet::GammaOp op);
  LogicalResult emitWcetPenalty(wcet::PenaltyOp op);
  LogicalResult emitSpechlsLut(spechls::LUTOp op);
  LogicalResult emitWcetInit(wcet::InitOp op);

  // Collect FSM info without emitting
  void collectFsm(spechls::FSMOp op);
  void collectPack(spechls::PackOp op);
};

// ---------------------------------------------------------------------------
// Name management
// ---------------------------------------------------------------------------

std::string WcetCppEmitter::newVar(StringRef prefix) { return (prefix + std::to_string(varCounter++)).str(); }

std::string WcetCppEmitter::nameOf(Value v) {
  auto it = valueNames.find(v);
  if (it != valueNames.end())
    return it->second;
  // Emit a diagnostic so the user knows which SSA value is unresolved,
  // then return a placeholder that will cause a compile error (intentional:
  // better a clear compile error than a silent wrong result).
  if (auto *defOp = v.getDefiningOp()) {
    defOp->emitWarning("wcet-cpp-export: unresolved SSA value used before "
                       "definition; op: ")
        << defOp->getName();
  } else {
    // Block argument not in valueNames — should have been registered during
    // signature parsing.
    llvm::errs() << "wcet-cpp-export: unresolved block argument\n";
  }
  // Return an expression that produces a known-unknown value of the right
  // type so the generated code compiles (the analysis result will be
  // conservative / fully-unknown for affected variables).
  Type t = v.getType();
  return mlirTypeReset(t);
}

// ---------------------------------------------------------------------------
// Top-level entry
// ---------------------------------------------------------------------------

LogicalResult WcetCppEmitter::translate(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](wcet::CoreOp coreOp) {
    if (failed(translateWcetCore(coreOp)))
      result = failure();
  });
  return result;
}

// ---------------------------------------------------------------------------
// Main per-wcet.core translator
// ---------------------------------------------------------------------------

LogicalResult WcetCppEmitter::translateWcetCore(wcet::CoreOp coreOp) {
  // Derive class name from symbol (capitalise first letter)
  std::string className = coreOp.getSymName().str();
  if (!className.empty())
    className[0] = std::toupper(className[0]);

  // ---- Parse input arguments ----
  int delayPos = 0;
  int instrPos = 0;
  unsigned totalArch = 0;
  for (unsigned i = 0; i < coreOp.getNumArguments(); ++i) {
    BlockArgument arg = coreOp.getArgument(i);

    // Skip state arguments
    if (isa<spechls::StructType>(arg.getType()))
      continue;

    ArgInfo info;
    info.mlirType = arg.getType();
    info.cppType = mlirTypeToCpp(arg.getType());

    unsigned archIdx = 0;
    auto attrs = coreOp.getArgAttrDict(i);
    if (attrs && attrs.get("wcet.instrNb")) {
      info.isInstr = true;
      info.name = "instr" + std::to_string(instrPos++);
    } else if (attrs && attrs.get("wcet.nbPred")) {
      info.isDelay = true;
      info.nbPred = (int)cast<IntegerAttr>(attrs.get("wcet.nbPred")).getValue().getSExtValue();
      info.delayPos = delayPos++;
      info.name = "delay" + std::to_string(info.delayPos);
    } else {
      for (auto &a : inArgs)
        if (!a.isInstr && !a.isDelay)
          ++archIdx;
      info.name = "arch" + std::to_string(archIdx);
      totalArch++;
    }

    // Register the SSA value so body ops can reference it as "in.fieldName"
    valueNames[arg] = "in." + info.name;
    inArgs.push_back(info);
  }

  // ---- Parse output types from function result types ----
  {
    unsigned currentArchIdx = 0, delayIdx = 0;
    for (auto t : coreOp.getFunctionType().getResults()) {
      if (isa<spechls::StructType>(t))
        continue;
      ArgInfo info;
      info.mlirType = t;
      info.cppType = mlirTypeToCpp(t);

      info.name = (currentArchIdx < totalArch) ? ("arch" + std::to_string(currentArchIdx++))
                                               : ("delay" + std::to_string(delayIdx++));
      outArgs.push_back(info);
    }
  }

  // ---- First pass: collect FSM/pack metadata ----
  coreOp.walk([&](spechls::FSMOp op) { collectFsm(op); });
  coreOp.walk([&](spechls::PackOp op) { collectPack(op); });

  // ---- Find wcet.commit ----
  wcet::CommitOp commitOp;
  coreOp.walk([&](wcet::CommitOp op) { commitOp = op; });
  if (!commitOp)
    return coreOp.emitError("no wcet.commit found in wcet.core");

  // Store the commit operands (skipping the trailing state value if present)
  for (unsigned i = 0; i < commitOp.getNumOperands(); ++i) {
    Value v = commitOp.getOperand(i);
    if (isa<spechls::StructType>(v.getType()))
      continue;
    commitOperands.push_back(v);
  }

  // ---- Emit ----
  emitFileHeader(className);
  emitInStateStruct();
  emitOutStateStruct();
  emitInFsmStruct();
  emitFsmFunction();
  emitCoreFunction(coreOp);
  if (version == Version::V2)
    emitCoreNextFunction(coreOp);
  emitSetupAnalysis();
  emitSetupNextState(commitOp);
  emitIsEqual();
  emitClearInState();
  emitClearOutState();
  emitPrintOutState();
  emitClassFooter(className);

  return success();
}

// ---------------------------------------------------------------------------
// File-level boilerplate
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitFileHeader(StringRef className) {
  os << "// Auto-generated by spechls-translate --export-wcet-cpp\n";
  os << "// DO NOT EDIT\n\n";
  os << "#include \"compilewcet.h\"\n";
  os << "#include \"abstract_ap_int.h\"\n";
  os << "#include \"spechls_support.h\"\n";
  os << "#include <map>\n";
  os << "#include <ostream>\n";
  os << "#include <unordered_set>\n";
  os << "#include <vector>\n\n";
  os << "class " << className << " : public CoreAnalysis {\n";
  os << "public:\n\n";
  os << "\tstd::vector<outState> coreAnalysis(const inState &in) override {\n";
  switch (version) {
  case Version::V1:
    os << "\t\tstd::vector<outState> outs = _core(*((_inState *)in));\n";
    os << "\t\treturn outs;\n";
    break;
  case Version::V2:
    os << "\t\t_inFSM fsmIn = _core(*((_inState *)in));\n";
    os << "\t\treturn _fsm(*((_inState *)in), fsmIn);\n";
  }
  os << "\t}\n\n";
  os << "\tunsigned int getPen(const outState &out) override {\n";
  os << "\t\treturn ((_outState *)out)->pen;\n";
  os << "\t}\n\n";
}

// ---------------------------------------------------------------------------
// _inState struct
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitInStateStruct() {
  os << "private:\n";
  os << "  struct _inState {\n";
  for (auto &a : inArgs)
    os << "    " << a.cppType << " " << a.name << ";\n";
  os << "  };\n\n";
}

// ---------------------------------------------------------------------------
// _outState struct
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitOutStateStruct() {
  os << "  struct _outState {\n";
  for (auto &a : outArgs)
    os << "    " << a.cppType << " " << a.name << ";\n";
  os << "    unsigned int pen;\n";
  os << "  };\n\n";
}

// ---------------------------------------------------------------------------
// _inFSM struct  (fields come from the spechls.pack operands)
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitInFsmStruct() {
  os << "  struct _inFSM {\n";
  for (unsigned i = 0; i < fsmInfo.packFieldNames.size(); ++i)
    os << "    " << fsmInfo.packFieldTypes[i] << " " << fsmInfo.packFieldNames[i] << ";\n";
  os << "  };\n\n";
}

// ---------------------------------------------------------------------------
// setupAnalysis  – all fields unknown, instr set to the argument
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitSetupAnalysis() {
  os << "public:\n"
        "  inState setupAnalysis(unsigned int instr) override {\n"
        "    _inState *res = new _inState{\n";
  bool first = true;
  for (auto &a : inArgs) {
    if (!first)
      os << ",\n";
    first = false;
    os << "      ";
    if (a.isInstr)
      os << "instr";
    else if (a.mlirType.isInteger(1))
      os << "abstract_bool(true)";
    else
      os << mlirTypeReset(a.mlirType); // delays/arch fields start unknown
  }
  os << "\n    };\n    return res;\n  }\n\n";
}

// ---------------------------------------------------------------------------
// setupNextState – map _outState fields back to _inState
// The wcet.commit operands define exactly the output values in order;
// we map them positionally to outArgs names.
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitSetupNextState(wcet::CommitOp commitOp) {
  os << "  inState setupNextState(const outState &out,\n"
        "                         unsigned int instr) override {\n"
        "    _outState o = *((_outState *)out);\n"
        "    _inState *res = new _inState{\n"
        "      instr";
  for (auto &a : outArgs)
    os << ",\n      o." << a.name;
  os << "\n    };\n    return res;\n  }\n\n";
}

// ---------------------------------------------------------------------------
// isEqual – field-by-field comparison
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitIsEqual() {
  os << "  bool isEqual(const inState &in1,\n"
        "               const inState &in2) override {\n"
        "    _inState i1 = *((_inState *)in1);\n"
        "    _inState i2 = *((_inState *)in2);\n"
        "    bool res = true;\n";

  for (auto &a : inArgs) {
    if (a.isInstr)
      continue; // instr is always equal by construction

    if (a.cppType.find("_tab") != std::string::npos) {
      // Sparse map comparison:
      // Two tabs are equal iff for every address present in either map,
      // both have the same unknown/value status.
      // An address absent from a map means its entry is unknown,
      // so:  absent == absent      → equal (both unknown)
      //       absent == known val  → not equal (unknown vs known)
      //       known  == known      → equal iff same value
      os << "    {\n"
         << "      // Tab isEqual via sparse maps\n"
         << "      const auto &m1 = i1." << a.name << ".entries();\n"
         << "      const auto &m2 = i2." << a.name
         << ".entries();\n"
         // Keys in m1 not unknown: check they match in m2
         << "      for (auto &kv : m1) {\n"
         << "        if (kv.second.unknown) continue; // unknown in m1\n"
         << "        auto it = m2.find(kv.first);\n"
         << "        if (it == m2.end() || it->second.unknown) return false; // m2 unknown\n"
         << "        if (kv.second.value != it->second.value) return false;\n"
         << "      }\n"
         // Keys in m2 not unknown: must exist and be known in m1
         << "      for (auto &kv : m2) {\n"
         << "        if (kv.second.unknown) continue;\n"
         << "        auto it = m1.find(kv.first);\n"
         << "        if (it == m1.end() || it->second.unknown) return false;\n"
         << "        if (kv.second.value != it->second.value) return false;\n"
         << "      }\n"
         << "    }\n";
    } else {
      // Scalar (abstract_bool or abstract_ap_uint<W>)
      os << "    if (i1." << a.name << ".unknown == i2." << a.name << ".unknown)\n"
         << "      res = res && (i1." << a.name << ".unknown ? true\n"
         << "                 : i1." << a.name << ".value == i2." << a.name << ".value);\n"
         << "    else return false;\n";
    }
  }
  os << "    return res;\n  }\n\n";
}

// ---------------------------------------------------------------------------
// clearInState / clearOutState
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitClearInState() {
  os << "  void clearInState(inState &in) override {\n"
        "    delete ((_inState *)in);\n"
        "  }\n\n";
}

void WcetCppEmitter::emitClearOutState() {
  os << "  void clearOutState(std::vector<outState> &outs) override {\n"
        "    for (auto o : outs)\n"
        "      delete ((_outState *)o);\n"
        "  }\n\n";
}

// ---------------------------------------------------------------------------
// printOutState
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitPrintOutState() {
  os << "  void printOutState(std::ostream &os,\n"
        "                     const outState &out) override {\n"
        "    _outState o = *((_outState *)out);\n"
        "    os << \"outState {\\n\";\n";
  for (auto &a : outArgs) {
    if (a.cppType.find("_tab") != std::string::npos) {
      // Print all known entries from the sparse map
      os << "    os << \"\\t" << a.name << " = {\\n\";\n"
         << "    for (auto &_kv : o." << a.name << ".entries())\n"
         << "      if (!_kv.second.unknown)\n"
         << "        os << \"\\t\\t[\" << _kv.first << \"] = \" << _kv.second << \"\\n\";\n"
         << "    os << \"\\t}\\n\";\n";
    } else {
      os << "    os << \"\\t" << a.name << " = \" << o." << a.name << " << \"\\n\";\n";
    }
  }
  os << "    os << \"\\tpen = \" << o.pen << \"\\n\";\n"
        "    os << \"}\\n\";\n"
        "  }\n\n";
}

// ---------------------------------------------------------------------------
// topoSort  -  Kahn's algorithm on the flat op list of a single block.
//
// MLIR SSA guarantees that every Value is defined before it is used within
// a well-formed program, but the wcet.core block we receive may have ops
// in an order that predates some lowering passes, leaving forward references.
// We sort ops explicitly by their use-def dependencies before emitting C++
// so that every variable is declared before it is used.
//
// Block arguments (in.*, constants already bound) are treated as always
// available.  Ops with no pending operands stay in their original relative
// order (stable property of Kahn's algorithm).
// ---------------------------------------------------------------------------

static std::vector<Operation *> topoSort(Block &block) {
  llvm::DenseSet<Value> available;
  for (auto arg : block.getArguments())
    available.insert(arg);

  llvm::DenseMap<Operation *, unsigned> pending;
  llvm::DenseMap<Value, llvm::SmallVector<Operation *, 4>> waiters;
  std::vector<Operation *> ready;
  std::vector<Operation *> sorted;

  for (auto &op : block) {
    unsigned count = 0;
    for (Value v : op.getOperands()) {
      if (!available.count(v)) {
        ++count;
        waiters[v].push_back(&op);
      }
    }
    pending[&op] = count;
    if (count == 0)
      ready.push_back(&op);
  }

  while (!ready.empty()) {
    // Take the first ready op to preserve original order among equals
    Operation *op = ready.front();
    ready.erase(ready.begin());
    sorted.push_back(op);

    for (Value result : op->getResults()) {
      available.insert(result);
      auto it = waiters.find(result);
      if (it == waiters.end())
        continue;
      for (Operation *waiter : it->second) {
        if (--pending[waiter] == 0)
          ready.push_back(waiter);
      }
    }
  }

  // Safety fallback: if cycles detected (shouldn't happen in SSA),
  // return original order.
  if (sorted.size() != (size_t)std::distance(block.begin(), block.end())) {
    sorted.clear();
    for (auto &op : block)
      sorted.push_back(&op);
  }

  return sorted;
}

// ---------------------------------------------------------------------------
// core() function  –  main body translation
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitCoreFunction(wcet::CoreOp coreOp) {
  switch (version) {
  case Version::V1:
    os << "\tstd::vector<outState> _core(_inState &in) {\n";
    break;
  case Version::V2:
    os << "\t_inFSM _core(_inState &in) {\n";
  }

  // Get the backward cone of all speculation ctrl signals

  std::vector<Operation *> ops;
  // get the FSM's ctrl
  // spechls::PackOp fsmCtrls;
  // coreOp->walk(
  //     [&](spechls::FSMOp fsm) { fsmCtrls = mlir::dyn_cast_or_null<spechls::PackOp>(fsm.getMispec().getDefiningOp());
  //     });
  // if (fsmCtrls == nullptr)
  //   return;
  //
  // // For each ctrl signals add ops
  // for (auto ctrl : fsmCtrls.getInputs()) {
  //   std::vector<mlir::Operation *> stack;
  //   stack.push_back(ctrl.getDefiningOp());
  //   while (!stack.empty()) {
  //     mlir::Operation *cOp = stack.back();
  //     stack.pop_back();
  //     if (std::find(ops.begin(), ops.end(), cOp) != ops.end())
  //       continue;
  //     ops.insert(ops.begin(), cOp);
  //     for (auto nOp : cOp->getOperands()) {
  //       if (nOp.getDefiningOp() != nullptr)
  //         stack.insert(stack.begin(), nOp.getDefiningOp());
  //     }
  //   }
  // }

  // Build logics to compute speculation's ctrl
  ops = topoSort(coreOp.getBody().front());
  for (Operation *op : ops) {
    if (failed(emitOp(op)))
      op->emitWarning("wcet-cpp-export: unhandled op ") << op->getName();
  }
  // Build baseOut from the wcet.commit operands (positionally mapped to outArgs)
  if (version == Version::V1) {
    os << "\n    // Build base output state from wcet.commit\n";
    os << "    _outState baseOut;\n";
    for (unsigned i = 0; i < outArgs.size() && i < commitOperands.size(); ++i)
      os << "    baseOut." << outArgs[i].name << " = " << nameOf(commitOperands[i]) << ";\n";
  }
  // Build _inFSM from the spechls.pack operands
  os << "\n    // Build FSM input from spechls.pack\n";
  os << "    _inFSM fsmIn;\n";
  for (unsigned i = 0; i < fsmInfo.packFieldNames.size() && i < fsmInfo.packOperandValues.size(); ++i)
    os << "    fsmIn." << fsmInfo.packFieldNames[i] << " = " << nameOf(fsmInfo.packOperandValues[i]) << ";\n";

  switch (version) {
  case Version::V1:
    os << "\n\t\treturn _fsm(baseOut, fsmIn);\n";
    break;
  case Version::V2:
    os << "\n\t\treturn fsmIn;\n";
  }

  os << "\t}\n\n";
}

void WcetCppEmitter::emitCoreNextFunction(wcet::CoreOp coreOp) {
  // Get ctrl signals
  spechls::PackOp ctrls = nullptr;
  coreOp->walk(
      [&](spechls::FSMOp fsm) { ctrls = mlir::dyn_cast_or_null<spechls::PackOp>(fsm.getMispec().getDefiningOp()); });

  if (ctrls == nullptr) {
    llvm::errs() << "ERROR: Couldn't find packOp\n";
    return;
  }

  os << "\t_outState _core_next(_inState &in";
  std::map<mlir::Operation *, std::string> ctrlOps;
  for (unsigned i = 0; i < ctrls->getNumOperands(); i++) {
    mlir::Operation *op = ctrls->getOperand(i).getDefiningOp();
    os << ", " << mlirTypeToCpp(op->getResultTypes().front()) << " " << valueNames[ctrls.getOperand(i)];
    ctrlOps[op] = valueNames[ctrls.getOperand(i)];
  }

  os << ") {\n";

  // Print function Body
  std::vector<Operation *> ops = topoSort(coreOp.getBody().front());
  for (Operation *op : ops) {
    if (ctrlOps.find(op) != ctrlOps.end()) {
      continue;
    }
    if (failed(emitOp(op)))
      op->emitWarning("wcet-cpp-export: unhandled op ") << op->getName();
  }

  // Print function footer
  os << "\t\t_outState baseOut;\n";
  for (unsigned i = 0; i < outArgs.size() && i < commitOperands.size(); ++i)
    os << "\t\tbaseOut." << outArgs[i].name << " = " << nameOf(commitOperands[i]) << ";\n";
  os << "\t\treturn baseOut;\n";
  os << "\t}\n";
}

// ---------------------------------------------------------------------------
// Per-operation emitters
// ---------------------------------------------------------------------------

LogicalResult WcetCppEmitter::emitOp(Operation *op) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<hw::ConstantOp>([&](auto o) { return emitConstant(o); })
      .Case<hw::BitcastOp>([&](auto o) { return emitHwBitcast(o); })
      .Case<comb::ConcatOp>([&](auto o) { return emitCombConcat(o); })
      .Case<comb::ExtractOp>([&](auto o) { return emitCombExtract(o); })
      .Case<comb::ICmpOp>([&](auto o) { return emitCombICmp(o); })
      .Case<comb::AndOp>([&](auto o) { return emitCombAnd(o); })
      .Case<comb::OrOp>([&](auto o) { return emitCombOr(o); })
      .Case<comb::MuxOp>([&](auto o) { return emitCombMux(o); })
      .Case<comb::AddOp>([&](auto o) { return emitCombAdd(o); })
      .Case<comb::MulOp>([&](auto o) { return emitCombMul(o); })
      .Case<comb::SubOp>([&](auto o) { return emitCombSub(o); })
      .Case<comb::DivUOp>([&](auto o) { return emitCombDivU(o); })
      .Case<comb::DivSOp>([&](auto o) { return emitCombDivS(o); })
      .Case<comb::ModUOp>([&](auto o) { return emitCombModU(o); })
      .Case<comb::ModSOp>([&](auto o) { return emitCombModS(o); })
      .Case<comb::XorOp>([&](auto o) { return emitCombXor(o); })
      .Case<comb::ShlOp>([&](auto o) { return emitCombShl(o); })
      .Case<comb::ShrUOp>([&](auto o) { return emitCombShrU(o); })
      .Case<comb::ShrSOp>([&](auto o) { return emitCombShrS(o); })
      .Case<comb::ParityOp>([&](auto o) { return emitCombParity(o); })
      .Case<comb::ReplicateOp>([&](auto o) { return emitCombReplicate(o); })
      .Case<comb::ReverseOp>([&](auto o) { return emitCombReverse(o); })
      // .Case<comb::TruthTableOp>([&](auto o) { return emitCombTruthTable(o); })
      .Case<spechls::LoadOp>([&](auto o) { return emitSpechlsLoad(o); })
      .Case<spechls::AlphaOp>([&](auto o) { return emitSpechlsAlpha(o); })
      .Case<spechls::LUTOp>([&](auto o) { return emitSpechlsLut(o); })
      .Case<wcet::GammaOp>([&](auto o) { return emitWcetGamma(o); })
      .Case<wcet::PenaltyOp>([&](auto o) { return emitWcetPenalty(o); })
      .Case<wcet::InitOp>([&](auto o) { return emitWcetInit(o); })
      // These are handled separately or ignored
      .Case<spechls::PackOp, spechls::FSMOp, wcet::CommitOp, wcet::InitOp>([&](auto) { return success(); })
      .Default([&](Operation *o) {
        // Unknown op: emit a comment
        os << "    /* unhandled: " << o->getName().getStringRef() << " */\n";
        return success();
      });
}

LogicalResult WcetCppEmitter::emitConstant(hw::ConstantOp op) {
  std::string var = newVar("c");
  valueNames[op.getResult()] = var;
  auto itype = cast<IntegerType>(op.getType());
  std::string ctype = mlirTypeToCpp(itype);
  int64_t val = op.getValue().getSExtValue();
  os << "    " << ctype << " " << var << " = " << ctype << "(" << val << ");\n";
  return success();
}

LogicalResult WcetCppEmitter::emitHwBitcast(hw::BitcastOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;

  Type srcType = op.getInput().getType();
  Type dstType = op.getType();
  std::string src = nameOf(op.getInput());

  // Helper lambdas to query type properties
  auto isI1 = [](Type t) {
    auto i = dyn_cast<IntegerType>(t);
    return i && i.getWidth() == 1;
  };
  // auto isBool = [](Type t) { return isa<IntegerType>(t) && cast<IntegerType>(t).getWidth() == 1; };
  // For our purposes i1 and bool map to the same abstract_bool in C++,
  // but we distinguish signed vs unsigned for wider types.
  auto isSigned = [](Type t) {
    if (auto i = dyn_cast<IntegerType>(t))
      return i.isSigned();
    return false;
  };
  auto width = [](Type t) -> unsigned {
    if (auto i = dyn_cast<IntegerType>(t))
      return i.getWidth();
    return 0;
  };

  std::string dstCtype = mlirTypeToCpp(dstType);
  std::string call;

  if (isI1(dstType)) {
    // destination is i1 / bool
    if (isI1(srcType))
      call = src; // noop
    else
      call = "bitcast_to_bool(" + src + ")";
  } else if (isI1(srcType)) {
    // source is bool → destination is a wider unsigned
    call = "bitcast_to_uint1(" + src + ")";
  } else if (isSigned(dstType)) {
    call = "bitcast_to_int<" + std::to_string(width(dstType)) + ">(" + src + ")";
  } else {
    call = "bitcast_to_uint<" + std::to_string(width(dstType)) + ">(" + src + ")";
  }

  os << "    " << dstCtype << " " << var << " = " << call << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombConcat(comb::ConcatOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;

  auto resType = cast<IntegerType>(op.getType());
  unsigned resWidth = resType.getWidth();
  std::string rtype = "abstract_ap_uint<" + std::to_string(resWidth) + ">";

  auto inputs = op.getInputs();
  unsigned n = inputs.size();

  // Compute the bit offset (shift amount) for each operand.
  // operand[i] is shifted left by sum of widths of operands[i+1..N-1].
  std::vector<unsigned> widths(n);
  for (unsigned i = 0; i < n; ++i)
    widths[i] = cast<IntegerType>(inputs[i].getType()).getWidth();

  // Build a cumulative suffix-sum of widths for shift amounts
  // shift[i] = widths[i+1] + ... + widths[N-1]
  std::vector<unsigned> shift(n, 0);
  for (int i = (int)n - 2; i >= 0; --i)
    shift[i] = shift[i + 1] + widths[i + 1];

  // Emit:
  //   rtype var = rtype(0);              // start with 0
  //   if (op0.unknown || op1.unknown...) var.unknown = true;
  //   else { var = (rtype(op0) << shift0) | (rtype(op1) << shift1) | ...; }
  os << "    " << rtype << " " << var << ";\n";

  // Unknown guard: if any operand is unknown → result is unknown
  os << "    if (";
  for (unsigned i = 0; i < n; ++i) {
    if (i)
      os << " || ";
    // abstract_bool has .unknown directly; other types too
    os << nameOf(inputs[i]) << ".unknown";
  }
  os << ") {\n";
  os << "      " << var << ".unknown = true;\n";
  os << "    } else {\n";

  // All operands are known: build the concatenated value
  os << "      " << var << ".unknown = false;\n";
  os << "      " << var << ".value = ";
  for (unsigned i = 0; i < n; ++i) {
    if (i)
      os << " | ";
    // Cast each operand to the result width, then shift to its position.
    // For abstract_bool we use the .value directly (it is a bool, cast to int).
    std::string operandName = nameOf(inputs[i]);
    if (widths[i] == 1) {
      // i1 / bool: treat as 1-bit uint
      os << "(ap_uint<" << resWidth << ">(" << operandName << ".value ? 1 : 0)"
         << " << " << shift[i] << "u)";
    } else {
      os << "(ap_uint<" << resWidth << ">(" << operandName << ".value)"
         << " << " << shift[i] << "u)";
    }
  }
  os << ";\n";
  os << "    }\n";

  return success();
}

LogicalResult WcetCppEmitter::emitCombExtract(comb::ExtractOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  auto srcType = cast<IntegerType>(op.getInput().getType());
  auto resType = cast<IntegerType>(op.getType());
  std::string srcCtype = mlirTypeToCpp(srcType);
  // unsigned srcWidth = srcType.getWidth();
  unsigned low = op.getLowBit();
  unsigned width = resType.getWidth();
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);

  // Build the shifted+masked expression in the source type first,
  // then convert to the destination type.
  //
  // Special case low==0: no shift needed.
  std::string shifted =
      (low == 0) ? nameOf(op.getInput()) : "(" + nameOf(op.getInput()) + " >> " + std::to_string(low) + ")";

  // The masked expression is always in the source type (e.g. uint<32>).
  std::string masked = shifted + " & " + srcCtype + "(" + std::to_string(mask) + ")";

  // Final conversion to the destination type.
  // abstract_bool cannot be constructed from abstract_ap_uint<W> via a
  // functional cast on all compilers (template deduction may fail), so
  // we use bitcast_to_bool() explicitly for i1 destinations.
  if (width == 1) {
    os << "    abstract_bool " << var << " = bitcast_to_bool(" << srcCtype << "(" << masked << "));\n";
  } else {
    std::string dstCtype = "abstract_ap_uint<" + std::to_string(width) + ">";
    os << "    " << dstCtype << " " << var << " = " << dstCtype << "(" << masked << ");\n";
  }
  return success();
}

LogicalResult WcetCppEmitter::emitCombICmp(comb::ICmpOp op) {
  std::string var = newVar("b");
  valueNames[op.getResult()] = var;
  std::string lhs = nameOf(op.getLhs());
  std::string rhs = nameOf(op.getRhs());
  std::string cmpOp;
  switch (op.getPredicate()) {
  case comb::ICmpPredicate::eq:
    cmpOp = "==";
    break;
  case comb::ICmpPredicate::ne:
    cmpOp = "!=";
    break;
  case comb::ICmpPredicate::ult:
  case comb::ICmpPredicate::slt:
    cmpOp = "<";
    break;
  case comb::ICmpPredicate::ugt:
  case comb::ICmpPredicate::sgt:
    cmpOp = ">";
    break;
  case comb::ICmpPredicate::ule:
  case comb::ICmpPredicate::sle:
    cmpOp = "<=";
    break;
  case comb::ICmpPredicate::uge:
  case comb::ICmpPredicate::sge:
    cmpOp = ">=";
    break;
  default:
    cmpOp = "/*cmp?*/==";
    break;
  }
  os << "    abstract_bool " << var << " = " << lhs << " " << cmpOp << " " << rhs << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombAnd(comb::AndOp op) {
  std::string var = newVar("b");
  valueNames[op.getResult()] = var;
  // comb.and on i1 → logical &&
  if (cast<IntegerType>(op.getType()).getWidth() == 1) {
    std::string expr = nameOf(op.getInputs()[0]);
    for (unsigned i = 1; i < op.getInputs().size(); ++i)
      expr += " && " + nameOf(op.getInputs()[i]);
    os << "    abstract_bool " << var << " = " << expr << ";\n";
  } else {
    std::string ctype = mlirTypeToCpp(op.getType());
    std::string expr = nameOf(op.getInputs()[0]);
    for (unsigned i = 1; i < op.getInputs().size(); ++i)
      expr += " & " + nameOf(op.getInputs()[i]);
    os << "    " << ctype << " " << var << " = " << expr << ";\n";
  }
  return success();
}

LogicalResult WcetCppEmitter::emitCombOr(comb::OrOp op) {
  std::string var = newVar("b");
  valueNames[op.getResult()] = var;
  if (cast<IntegerType>(op.getType()).getWidth() == 1) {
    std::string expr = nameOf(op.getInputs()[0]);
    for (unsigned i = 1; i < op.getInputs().size(); ++i)
      expr += " || " + nameOf(op.getInputs()[i]);
    os << "    abstract_bool " << var << " = " << expr << ";\n";
  } else {
    std::string ctype = mlirTypeToCpp(op.getType());
    std::string expr = nameOf(op.getInputs()[0]);
    for (unsigned i = 1; i < op.getInputs().size(); ++i)
      expr += " | " + nameOf(op.getInputs()[i]);
    os << "    " << ctype << " " << var << " = " << expr << ";\n";
  }
  return success();
}

LogicalResult WcetCppEmitter::emitCombMux(comb::MuxOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string cond = nameOf(op.getCond());
  std::string tval = nameOf(op.getTrueValue());
  std::string fval = nameOf(op.getFalseValue());
  // Emit: T var = cond.unknown ? T() : (cond.value ? tval : fval);
  os << "    " << ctype << " " << var << " = " << cond << ".unknown ? " << ctype << "() : (" << cond << ".value ? "
     << tval << " : " << fval << ");\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombAdd(comb::AddOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string expr = nameOf(op.getInputs()[0]);
  for (unsigned i = 1; i < op.getInputs().size(); ++i)
    expr += " + " + nameOf(op.getInputs()[i]);
  os << "    " << ctype << " " << var << " = " << expr << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombMul(comb::MulOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string expr = nameOf(op.getInputs()[0]);
  for (unsigned i = 1; i < op.getInputs().size(); ++i)
    expr += " * " + nameOf(op.getInputs()[i]);
  os << "    " << ctype << " " << var << " = " << expr << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitSpechlsLoad(spechls::LoadOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string arr = nameOf(op.getArray());
  std::string idx = nameOf(op.getIndex());
  os << "    " << ctype << " " << var << " = " << arr << "[" << idx << "];\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombSub(comb::SubOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  os << "    " << ctype << " " << var << " = " << nameOf(op.getLhs()) << " - " << nameOf(op.getRhs()) << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombDivU(comb::DivUOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  os << "    " << ctype << " " << var << " = " << nameOf(op.getLhs()) << " / " << nameOf(op.getRhs()) << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombDivS(comb::DivSOp op) {
  // Signed division: cast both operands to the signed variant, divide,
  // then cast back. The result width is the same as the operands.
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  auto itype = cast<IntegerType>(op.getType());
  unsigned w = itype.getWidth();
  std::string uctype = "abstract_ap_uint<" + std::to_string(w) + ">";
  std::string sctype = "abstract_ap_int<" + std::to_string(w) + ">";
  // Cast to signed, divide, cast result back to unsigned (same bits).
  os << "    " << uctype << " " << var << " = " << uctype << "(" << sctype << "(" << nameOf(op.getLhs()) << ") / "
     << sctype << "(" << nameOf(op.getRhs()) << "));\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombModU(comb::ModUOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  os << "    " << ctype << " " << var << " = " << nameOf(op.getLhs()) << " % " << nameOf(op.getRhs()) << ";\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombModS(comb::ModSOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  auto itype = cast<IntegerType>(op.getType());
  unsigned w = itype.getWidth();
  std::string uctype = "abstract_ap_uint<" + std::to_string(w) + ">";
  std::string sctype = "abstract_ap_int<" + std::to_string(w) + ">";
  os << "    " << uctype << " " << var << " = " << uctype << "(" << sctype << "(" << nameOf(op.getLhs()) << ") % "
     << sctype << "(" << nameOf(op.getRhs()) << "));\n";
  return success();
}

LogicalResult WcetCppEmitter::emitCombXor(comb::XorOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  bool isBool = cast<IntegerType>(op.getType()).getWidth() == 1;
  std::string ctype = isBool ? "abstract_bool" : mlirTypeToCpp(op.getType());
  std::string expr = nameOf(op.getInputs()[0]);
  for (unsigned i = 1; i < op.getInputs().size(); ++i)
    expr += " ^ " + nameOf(op.getInputs()[i]);
  os << "    " << ctype << " " << var << " = " << expr << ";\n";
  return success();
}

// comb.shl  %lhs, %rhs  →  lhs << rhs  (logical left shift)
LogicalResult WcetCppEmitter::emitCombShl(comb::ShlOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  os << "    " << ctype << " " << var << " = " << nameOf(op.getLhs()) << " << " << nameOf(op.getRhs()) << ";\n";
  return success();
}

// comb.shru %lhs, %rhs  →  lhs >> rhs  (logical right shift, unsigned)
LogicalResult WcetCppEmitter::emitCombShrU(comb::ShrUOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  os << "    " << ctype << " " << var << " = " << nameOf(op.getLhs()) << " >> " << nameOf(op.getRhs()) << ";\n";
  return success();
}

// comb.shrs %lhs, %rhs  →  arithmetic right shift (signed)
// Cast to signed, shift, cast back.
LogicalResult WcetCppEmitter::emitCombShrS(comb::ShrSOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  auto itype = cast<IntegerType>(op.getType());
  unsigned w = itype.getWidth();
  std::string uctype = "abstract_ap_uint<" + std::to_string(w) + ">";
  std::string sctype = "abstract_ap_int<" + std::to_string(w) + ">";
  os << "    " << uctype << " " << var << " = " << uctype << "(" << sctype << "(" << nameOf(op.getLhs()) << ") >> "
     << nameOf(op.getRhs()) << ");\n";
  return success();
}

// comb.parity %x  →  XOR-reduction of all bits → abstract_bool
// If x is unknown, result is unknown.
// Otherwise: parity = (x.value.xor_reduce()) which is ap_uint<W>::xor_reduce()
// ap_uint provides .xor_reduce() returning bool (ap_uint<1>).
LogicalResult WcetCppEmitter::emitCombParity(comb::ParityOp op) {
  std::string var = newVar("b");
  valueNames[op.getResult()] = var;
  std::string src = nameOf(op.getInput());
  os << "    abstract_bool " << var << " = " << src << ".unknown ? "
     << "abstract_bool() : abstract_bool((bool)" << src << ".value.xor_reduce());\n";
  return success();
}

// comb.replicate %x  →  concatenate %x with itself N times to fill result width
// result_width = input_width * N  (N derived from widths)
LogicalResult WcetCppEmitter::emitCombReplicate(comb::ReplicateOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  auto srcType = cast<IntegerType>(op.getInput().getType());
  auto dstType = cast<IntegerType>(op.getType());
  unsigned srcW = srcType.getWidth();
  unsigned dstW = dstType.getWidth();
  unsigned n = dstW / srcW; // number of repetitions
  std::string rtype = "abstract_ap_uint<" + std::to_string(dstW) + ">";
  std::string src = nameOf(op.getInput());

  os << "    " << rtype << " " << var << ";\n";
  os << "    if (" << src << ".unknown) {\n";
  os << "      " << var << ".unknown = true;\n";
  os << "    } else {\n";
  os << "      " << var << ".unknown = false;\n";
  os << "      " << var << ".value = ";
  for (unsigned i = 0; i < n; ++i) {
    if (i)
      os << " | ";
    unsigned shift = (n - 1 - i) * srcW;
    os << "(ap_uint<" << dstW << ">(" << src << ".value) << " << shift << "u)";
  }
  os << ";\n";
  os << "    }\n";
  return success();
}

// comb.reverse %x  →  bit-reverse of %x
// ap_uint does not have a built-in reverse; we build it bit by bit.
LogicalResult WcetCppEmitter::emitCombReverse(comb::ReverseOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  auto itype = cast<IntegerType>(op.getType());
  unsigned w = itype.getWidth();
  std::string ctype = "abstract_ap_uint<" + std::to_string(w) + ">";
  std::string src = nameOf(op.getInput());

  os << "    " << ctype << " " << var << ";\n";
  os << "    if (" << src << ".unknown) {\n";
  os << "      " << var << ".unknown = true;\n";
  os << "    } else {\n";
  os << "      " << var << ".unknown = false;\n";
  os << "      " << var << ".value = 0;\n";
  os << "      for (unsigned _i = 0; _i < " << w << "u; ++_i)\n";
  os << "        " << var << ".value |= ap_uint<" << w << ">"
     << "((" << src << ".value >> _i) & 1) << (" << w << "u - 1u - _i);\n";
  os << "    }\n";
  return success();
}

// comb.truth_table %inputs  →  lookup into a boolean truth table
// The truth_table op has a lookup table attribute (an integer whose bits
// are the output values) and a variadic list of 1-bit inputs.
// The inputs form the index (input[0] is LSB of the index).
// If any input is unknown, result is unknown.
// LogicalResult WcetCppEmitter::emitCombTruthTable(comb::TruthTableOp op) {
//   std::string var = newVar("b");
//   valueNames[op.getResult()] = var;
//
//   // The truth table is stored as an integer attribute.
//   // The number of inputs determines the table size (2^N entries).
//   auto inputs = op.getInputs();
//   unsigned n = inputs.size();
//   // Retrieve the lookup table value as a uint64
//   uint64_t table = op.getLookupTable().getZExtValue();
//
//   os << "    abstract_bool " << var << ";\n";
//   // Unknown guard
//   os << "    if (";
//   for (unsigned i = 0; i < n; ++i) {
//     if (i)
//       os << " || ";
//     os << nameOf(inputs[i]) << ".unknown";
//   }
//   os << ") {\n";
//   os << "      " << var << ".unknown = true;\n";
//   os << "    } else {\n";
//   // Build the index from input bits
//   os << "      unsigned _idx = 0;\n";
//   for (unsigned i = 0; i < n; ++i)
//     os << "      _idx |= (unsigned(" << nameOf(inputs[i]) << ".value) << " << i << "u);\n";
//   os << "      " << var << ".unknown = false;\n";
//   os << "      " << var << ".value   = ((" << table << "ULL >> _idx) & 1) != 0;\n";
//   os << "    }\n";
//   return success();
// }

LogicalResult WcetCppEmitter::emitSpechlsAlpha(spechls::AlphaOp op) {
  // Condition is ignored (always true in our use case).
  // Emit: arr[idx] = val;  (returns the modified array as a new variable)
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string arr = nameOf(op.getArray());
  std::string idx = nameOf(op.getIndex());
  std::string val = nameOf(op.getValue());
  // Copy the array then write
  os << "    " << ctype << " " << var << " = " << arr << ";\n";
  os << "    " << var << "[" << idx << "] = " << val << ";\n";
  return success();
}

static void emitLutBody(raw_ostream &os, const std::string &var, const std::string &ctype, const std::string &idxName,
                        llvm::ArrayRef<int64_t> contents, unsigned resultWidth) {
  // Emit the table as a local static array of uint64_t.
  std::string tname = var + "_table";
  os << "    static const uint64_t " << tname << "[" << contents.size() << "] = {";
  for (unsigned i = 0; i < contents.size(); ++i) {
    if (i)
      os << ", ";
    // Mask to result width to avoid sign-extension surprises.
    uint64_t mask = (resultWidth == 64) ? ~0ULL : ((1ULL << resultWidth) - 1);
    os << ((uint64_t)contents[i] & mask) << "ULL";
  }
  os << "};\n";

  os << "    " << ctype << " " << var << ";\n";
  os << "    if (" << idxName << ".unknown) {\n";
  os << "      " << var << ".unknown = true;\n";
  os << "    } else {\n";
  os << "      " << var << ".unknown = false;\n";
  // Guard against out-of-range index.
  os << "      unsigned _lutIdx = (unsigned)" << idxName << ".value;\n";
  os << "      if (_lutIdx < " << contents.size() << "u)\n";
  // Assign directly from the raw uint64_t table entry into the ap_uint .value
  // field to avoid ambiguous constructor overload resolution.
  os << "        " << var << ".value = ap_uint<" << resultWidth << ">"
     << "((uint64_t)" << tname << "[_lutIdx]);\n";
  os << "      else\n";
  os << "        " << var << ".unknown = true; // out-of-range\n";
  os << "    }\n";
}

LogicalResult WcetCppEmitter::emitWcetInit(wcet::InitOp op) {
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  os << "    " << ctype << " " << var << "; // wcet.init (unknown)\n";
  return success();
}

LogicalResult WcetCppEmitter::emitSpechlsLut(spechls::LUTOp op) {
  std::string var = newVar("lut");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  unsigned rWidth = cast<IntegerType>(op.getType()).getWidth();
  emitLutBody(os, var, ctype, nameOf(op.getIndex()), op.getContents(), rWidth);
  return success();
}

LogicalResult WcetCppEmitter::emitWcetPenalty(wcet::PenaltyOp op) {
  // wcet.penalty %x by N  →  just an alias; the penalty is consumed by
  // the FSM gamma table. We create a variable that is identical to the
  // input value (the penalty is tracked separately in fsmInfo).
  std::string var = newVar("v");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string src = nameOf(op.getInput());
  os << "    " << ctype << " " << var << " = " << src << "; "
     << "// wcet.penalty by " << op.getDepth() << "\n";
  return success();
}

LogicalResult WcetCppEmitter::emitWcetGamma(wcet::GammaOp op) {
  std::string var = newVar("gamma");
  valueNames[op.getResult()] = var;
  std::string ctype = mlirTypeToCpp(op.getType());
  std::string sel = nameOf(op.getSelect());

  // gamma<T>(sel, v0, v1, ...) with unknown handling:
  // T var = sel.unknown ? T() : gamma<T>(sel.value, v0, v1, ...);
  std::string args = sel + ".value";
  for (auto operand : op.getInputs())
    args += ", " + nameOf(operand);

  os << "    " << ctype << " " << var << " = " << sel << ".unknown ? " << ctype;
  if (spechls::ArrayType at = dyn_cast_or_null<spechls::ArrayType>(op.getType())) {
    os << "(" << at.getSize() << ")";
  } else
    os << "()";
  os << " : gamma<" << ctype << ">(" << args << ");\n";

  return success();
}

// ---------------------------------------------------------------------------
// FSM / Pack collection (first pass, no emission)
// ---------------------------------------------------------------------------

void WcetCppEmitter::collectPack(spechls::PackOp op) {
  // Each operand of the pack becomes a field of _inFSM.
  // We use the FSM gamma names (in declaration order) as field names.
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Value v = op.getOperand(i);
    // Field name: use the corresponding gamma name if available, else "fieldN"
    std::string fname = (i < fsmInfo.gammaNames.size()) ? fsmInfo.gammaNames[i] : ("field" + std::to_string(i));
    fsmInfo.packFieldNames.push_back(fname);
    fsmInfo.packFieldTypes.push_back(mlirTypeToCpp(v.getType()));
    fsmInfo.packOperandValues.push_back(v);
    // Register the pack operand so it can be referenced as "in.<name>" in _fsm
    valueNames[v] = "in." + fname;
  }
}

void WcetCppEmitter::collectFsm(spechls::FSMOp op) {
  for (auto attr : op.getGammaNames()) {
    std::string gammaName = cast<StringAttr>(attr).str();
    fsmInfo.gammaNames.push_back(gammaName);
  }

  for (auto tableAttr : op.getInputDelaysAttr()) {
    std::vector<unsigned> table;
    for (auto v : cast<ArrayAttr>(tableAttr))
      table.push_back((unsigned)cast<IntegerAttr>(v).getInt());
    fsmInfo.penTables.push_back(table);
  }
}

// ---------------------------------------------------------------------------
// create() factory
// ---------------------------------------------------------------------------

void WcetCppEmitter::emitClassFooter(StringRef className) {
  os << "}; // class " << className
     << "\n\n"
        "extern \"C\" {\n"
        "CoreAnalysis *create() { return new "
     << className
     << "(); }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// fsm() function
// ---------------------------------------------------------------------------
void WcetCppEmitter::emitFsmFunction() {
  switch (version) {
  case Version::V2:
    os << "\tstd::vector<outState> _fsm(_inState inState, _inFSM in) {\n";
    break;
  case Version::V1:
    os << "\tstd::vector<outState> _fsm(_outState baseOut, _inFSM in) {\n";
  }

  // --- Penalty tables ---
  for (unsigned g = 0; g < fsmInfo.gammaNames.size(); ++g) {
    auto &pens = fsmInfo.penTables[g];
    os << "\t\tunsigned int pen_" << fsmInfo.gammaNames[g] << "[" << pens.size() << "] = {";
    for (unsigned i = 0; i < pens.size(); ++i) {
      if (i)
        os << ", ";
      os << pens[i];
    }
    os << "};\n";
  }
  os << "\n";

  // --- Per-gamma candidate sets ---
  for (unsigned g = 0; g < fsmInfo.gammaNames.size(); ++g) {
    auto &gname = fsmInfo.gammaNames[g];
    unsigned n = fsmInfo.penTables[g].size();
    switch (version) {
    case Version::V2:
      os << "\t\tstd::map<unsigned int, unsigned int> pens_" << gname << ";\n";
      os << "\t\tif (!in." << gname << ".unknown) {\n";
      os << "\t\t  pens_" << gname << "[in." << gname << ".value] = pen_" << gname << "[in." << gname << ".value];\n";
      os << "\t\t} else {\n";
      os << "\t\t  for (unsigned int i = 0; i < " << n << "; i++)\n";
      os << "\t\t    pens_" << gname << "[i] = pen_" << gname << "[i];\n";
      os << "\t\t}\n\n";
      break;
    case Version::V1:
      os << "\t\tstd::vector<unsigned int> pens_" << gname << ";\n";
      os << "\t\tif (!in." << gname << ".unknown) {\n";
      os << "\t\t  pens_" << gname << ".push_back(pen_" << gname << "[in." << gname << ".value]);\n";
      os << "\t\t} else {\n";
      os << "\t\t  for (unsigned int i = 0; i < " << n << "; i++)\n";
      os << "\t\t    pens_" << gname << ".push_back(pen_" << gname << "[i]);\n";
      os << "\t\t}\n\n";
    }
  }

  // --- Cartesian product -> unique penalty set ---
  if (Version::V1 == version) {
    os << "\t\tstd::unordered_set<unsigned int> pens;\n";
    for (unsigned g = 0; g < fsmInfo.gammaNames.size(); ++g)
      os << "\t\tfor (unsigned int p" << g << " : pens_" << fsmInfo.gammaNames[g] << ") {\n";
    os << "\t\t\tpens.insert(";
    for (unsigned g = 0; g < fsmInfo.gammaNames.size(); ++g) {
      if (g)
        os << " + ";
      os << "p" << g;
    }
    os << ");\n";
    for (unsigned g = 0; g < fsmInfo.gammaNames.size(); ++g)
      os << "\t\t}\n";
    os << "\n";
  }

  std::vector<const ArgInfo *> sortedDelayIns;
  for (auto &a : inArgs)
    if (a.isDelay)
      sortedDelayIns.push_back(&a);

  // --- Build one _outState per unique penalty ---
  os << "    std::vector<outState> res;\n";
  switch (version) {
  case Version::V1:
    os << "\t\tfor (unsigned int p : pens) {\n";
    break;
  case Version::V2:
    for (unsigned g = 0; g < fsmInfo.gammaNames.size(); g++) {
      os << "   for(std::pair<unsigned int, unsigned int> " << fsmInfo.gammaNames[g].substr(0, 3) << " : pens_"
         << fsmInfo.gammaNames[g] << ") {\n";
    }

    os << "unsigned int p = " << fsmInfo.gammaNames[0].substr(0, 3) << ".second";
    for (unsigned g = 1; g < fsmInfo.gammaNames.size(); g++) {
      os << " + " << fsmInfo.gammaNames[g].substr(0, 3) << ".second";
    }
    os << ";\n";
    os << "_outState baseOut = _core_next(inState";
    for (unsigned g = 0; g < fsmInfo.gammaNames.size(); g++) {
      os << ", " << fsmInfo.gammaNames[g].substr(0, 3) << ".first";
    }
    os << ");\n";
  }

  os << "\t\t\t_outState *out = new _outState();\n";

  unsigned delayOutIdx = 0;
  for (auto &out : outArgs) {
    bool isDelayOut = (out.name.rfind("delay", 0) == 0);

    if (!isDelayOut) {
      // Non-delay field: always copy from baseOut
      os << "      out->" << out.name << " = baseOut." << out.name << ";\n";
      continue;
    }

    // This output delay corresponds to the delay input at the same position
    if (delayOutIdx >= sortedDelayIns.size()) {
      // More outputs than inputs — safety fallback
      os << "      out->" << out.name << " = baseOut." << out.name << ";\n";
      ++delayOutIdx;
      continue;
    }

    const ArgInfo *correspondingIn = sortedDelayIns[delayOutIdx];
    int nbPred = correspondingIn->nbPred;

    os << "      // " << out.name << " (nbPred=" << nbPred << ")\n";
    if (nbPred == 0) {
      os << "      out->" << out.name << " = baseOut." << out.name << ";\n";
    } else {
      // p >= nbPred  → flush this stage
      // p <= nbPred → propagate value from the nbPred - p delays that is one
      //                step earlier
      os << "      if (p >= " << nbPred << ") {\n";

      if (auto itype = dyn_cast<IntegerType>(out.mlirType)) {
        if (itype.getWidth() == 1)
          os << "        out->" << out.name << " = abstract_bool(false);\n";
        else
          os << "        out->" << out.name << " = " << mlirTypeReset(out.mlirType) << ";\n";
      } else
        os << "        out->" << out.name << " = " << mlirTypeReset(out.mlirType) << ";\n";
      os << "      }\n";
      for (int i = 1; i < nbPred; i++) {
        os << "      else if (p == " << i << ") {\n";
        os << "        out->" << out.name << " = " << "baseOut." << sortedDelayIns[delayOutIdx - (nbPred - i)]->name
           << ";\n";
        os << "      }\n";
      }
      // Propagate from the previous delay input (one stage earlier)
      os << "      else {\n";
      os << "        out->" << out.name << " = baseOut." << out.name << ";\n";
      os << "      }\n";
    }
    ++delayOutIdx;
  }

  os << "out->pen = p;\n"
     << "res.push_back(out);\n";

  switch (version) {
  case Version::V1:
    os << "\t\t}\n\n";
    break;
  case Version::V2:
    for (unsigned g = 0; g < fsmInfo.gammaNames.size(); g++) {
      os << "}\n";
    }
  }

  os << "\t\treturn res;\n"
        "\t}\n\n";
}

} // namespace

// ---------------------------------------------------------------------------
// Public API + registration
// ---------------------------------------------------------------------------

namespace spechls {

mlir::LogicalResult exportWcetCpp(mlir::ModuleOp module, llvm::raw_ostream &os, WcetTranslateOptions options) {

  WcetCppEmitter emitter(os, std::move(options));
  return emitter.translate(module);
}

void registerExportWcetCpp() {
  static llvm::cl::opt<int> selectVersion("translate-version",
                                          llvm::cl::desc("Select the export version (default V2)."), llvm::cl::init(2));
  mlir::TranslateFromMLIRRegistration reg(
      "export-wcet-cpp", "Translate wcet.core to a C++ CoreAnalysis implementation",
      [](ModuleOp module, raw_ostream &os) { return exportWcetCpp(module, os, {selectVersion}); },

      [](mlir::DialectRegistry &registry) {
        // Register all dialects we consume
        registry.insert<hw::HWDialect, comb::CombDialect, spechls::SpecHLSDialect, wcet::WcetDialect>();
      });
}

} // namespace spechls
