//===- ExportUclid.cpp - SpecHLS → Uclid5 code generator --------*- C++ -*-===//
//
// Part of the SpecHLS project.
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Uclid5 code generator for the SpecHLS MLIR dialect.
//
// Mapping principles (high level):
//  * One `spechls.kernel` → one Uclid `module`.
//  * Kernel block arguments → `input` declarations (open module).
//  * Any state carried across iterations (all `spechls.mu`) → `var`.
//  * Delays can optionally be modelled as shift-register state (`var d_sK`).
//  * All combinational work of ONE C loop iteration is outlined into a single
//    `procedure step(...) returns (...)`.
//  * Each transition (`next { ... }`) performs exactly one loop iteration by
//    calling `step`, assigning primed state on the LHS of the call, and then
//    computing auxiliary primed assignments (e.g., delay stages).
//  * gamma/mux are mapped to reusable `define` functions for readability.
//
// These choices preserve the 1:1 mapping "one C loop iteration == one Uclid
// next-step" while keeping the transition relation explicit and analyzable.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Support/FileUtilities.h"
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/IndentedOstream.h>



#include <algorithm>
#include <stack>
#include <string>

using namespace mlir;

namespace spechls {

struct TranslationToUclidOptions {
  bool arraysWithBvIndex = true;      // model arrays as array [bvK] of T
  bool declareDelaysAsState = true;   // model DelayOp as shift-register state
};

//------------------------------------------------------------------------------
// Utilities for name mangling and small formatting helpers
//------------------------------------------------------------------------------

static inline std::string sanitize(StringRef s) {
  std::string t = s.str();
  for (char &c : t) {
    if (!(llvm::isAlnum(c) || c == '_')) c = '_';
  }
  return t;
}

//------------------------------------------------------------------------------
// UclidEmitter: stateful printer with scoped SSA name mapping
//------------------------------------------------------------------------------

class UclidEmitter {
  raw_indented_ostream os;

  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  ValueMapper valueMapper;
  std::stack<int64_t> valueInScopeCount;

  // Preamble collects reusable `define` helpers (gamma/mux variants).
  std::string preamble;

  struct GammaKey {
    unsigned arity;
    unsigned elemBW;  // 1 for boolean, else bv width
    unsigned selBW;   // for gammaN, 0 for mux2 boolean cond
    bool isBool;
    bool operator==(const GammaKey &o) const {
      return arity==o.arity && elemBW==o.elemBW && selBW==o.selBW && isBool==o.isBool;
    }
  };
  struct GammaKeyHash {
    // Hash pour GammaKey
    static unsigned getHashValue(const GammaKey &K) noexcept {
      return static_cast<unsigned>(llvm::hash_value(K.arity*31u + K.elemBW*17u + K.selBW*13u + (K.isBool ? 7u : 0u)));
    }

    // Comparaison d'égalité
    static bool isEqual(const GammaKey &L, const GammaKey &R) noexcept {
      return L == R;
    }

    // Valeurs sentinelles : choisir des valeurs impossibles pour des clés valides
    static GammaKey getEmptyKey() noexcept {
      return GammaKey{std::numeric_limits<uint32_t>::max()};
    }
    static GammaKey getTombstoneKey() noexcept {
      return GammaKey{std::numeric_limits<uint32_t>::max() - 1};
    }
  };

  llvm::DenseSet<GammaKey, GammaKeyHash> gammaDefs;

  // Delay op → vector of state stage variable names (s0..sD-1), and a temp name
  // for the newly computed input value in `step`.
  llvm::DenseMap<Operation*, SmallVector<std::string>> delayStageNames;
  llvm::DenseMap<Operation*, std::string> delayStepInputName;

  TranslationToUclidOptions opts;

public:
  explicit UclidEmitter(raw_ostream &ros, TranslationToUclidOptions options)
      : os(ros), opts(std::move(options)) {
    valueInScopeCount.push(0);
  }

  raw_indented_ostream &ostream() { return os; }

  //---- scoped mapping --------------------------------------------------------
  class Scope {
    llvm::ScopedHashTableScope<Value, std::string> scope;
    UclidEmitter &E;
  public:
    explicit Scope(UclidEmitter &e) : scope(e.valueMapper), E(e) {
      E.valueInScopeCount.push(E.valueInScopeCount.top());
    }
    ~Scope() { E.valueInScopeCount.pop(); }
  };

  bool has(Value v) const { return valueMapper.count(v); }
  StringRef getOrCreate(Value v, StringRef hintPrefix = "v") {
    if (!valueMapper.count(v)) {
      valueMapper.insert(v, llvm::formatv("{0}_{1}", hintPrefix, ++valueInScopeCount.top()));
    }
    return *valueMapper.begin(v);
  }

  //---- preamble helpers ------------------------------------------------------
  void appendPreamble(StringRef s) {
    preamble.append(s.begin(), s.end());
    if (preamble.empty() || preamble.back() != '\n') preamble.push_back('\n');
  }

  std::string ensureGammaDefine(unsigned arity, unsigned elemBW, unsigned selBW, bool isBool) {
    GammaKey key{arity, elemBW ? elemBW : 1u, selBW, isBool};
    if (gammaDefs.contains(key)) {
      return (llvm::Twine("gamma").concat(std::to_string(arity))
              .concat("_w").concat(std::to_string(isBool ? 1u : elemBW))).str();
    }
    gammaDefs.insert(key);

    std::string name = (llvm::Twine("gamma").concat(std::to_string(arity))
                        .concat("_w").concat(std::to_string(isBool ? 1u : elemBW))).str();

    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "define " << name << "(";
    if (arity == 2 && selBW == 0) {
      ss << "c: boolean";
    } else {
      ss << "s: bv" << selBW;
    }
    for (unsigned i=0;i<arity;++i) {
      ss << ", a" << i << ": " << (isBool ? "boolean" : ("bv" + std::to_string(elemBW)));
    }
    ss << ") : " << (isBool ? "boolean" : ("bv" + std::to_string(elemBW))) << " = ";

    if (arity == 2 && selBW == 0) {
      ss << "ite(c, a0, a1);\n";
    } else {
      ss << "case s of\n";
      for (unsigned i=0;i<arity;++i) ss << "  " << i << ": a" << i << ";\n";
      ss << "  otherwise: a0;\n";
      ss << "esac;\n";
    }
    ss.flush();
    appendPreamble(s);
    return name;
  }

  //---- printing primitives ---------------------------------------------------

  LogicalResult emitType(Location loc, Type t) {
    if (!t) { os << "void"; return success(); }
    if (auto it = dyn_cast<IntegerType>(t)) {
      if (it.getWidth() == 1) { os << "boolean"; return success(); }
      os << "bv" << it.getWidth();
      return success();
    }
    if (auto at = dyn_cast<spechls::ArrayType>(t)) {
      os << "array [";
      // Index type
      if (opts.arraysWithBvIndex) {
        // Round up width to cover [0..size-1].
        uint64_t size = at.getSize();
        unsigned idxW = size <= 1 ? 1u : llvm::Log2_64_Ceil(size);
        os << "bv" << idxW;
      } else {
        os << "integer";
      }
      os << "] of ";
      if (failed(emitType(loc, at.getElementType()))) return failure();
      return success();
    }
    // Fallback: named/opaque -> print as-is name
    if (auto st = dyn_cast<spechls::StructType>(t)) {
      os << sanitize(st.getName());
      return success();
    }
    return emitError(loc, "cannot emit Uclid type for ") << t;
  }

  LogicalResult emitOperand(Value v) {
    if (!v) return failure();
    if (auto *def = v.getDefiningOp()) {
      if (auto c = dyn_cast<circt::hw::ConstantOp>(def)) {
        auto ap = c.getValue();
        if (ap.getBitWidth() == 1) { os << (ap.isOne() ? "true" : "false"); return success(); }
        // Print as decimal; type carries the width.
        os << ap;
        return success();
      }
    }
    os << getOrCreate(v);
    return success();
  }

  // Generic binary op (bitvectors)
  LogicalResult printBinary(Operation *op, StringRef sym) {
    if (op->getNumResults() != 1) return op->emitOpError("expected single result");
    os << getOrCreate(op->getResult(0)) << " = ";
    if (failed(emitOperand(op->getOperand(0)))) return failure();
    os << " " << sym << " ";
    if (failed(emitOperand(op->getOperand(1)))) return failure();
    os << ";\n";
    return success();
  }

  //---- Delay helpers ---------------------------------------------------------
  std::string getDelayStageName(spechls::DelayOp d, unsigned s) {
    auto &vec = delayStageNames[d.getOperation()];
    if (vec.empty()) {
      // initialize with depth names
      unsigned depth = d.getDepth();
      vec.reserve(depth);
      std::string base = sanitize(getOrCreate(d.getResult()).str());
      for (unsigned i=0;i<depth;++i) vec.push_back(base + "_s" + std::to_string(i));
    }
    return vec[s];
  }

  void declareDelayVarsIfNeeded(spechls::KernelOp kernel) {
    if (!opts.declareDelaysAsState) return;
    kernel->walk([&](spechls::DelayOp d){
      (void)getDelayStageName(d, 0); // force init vector
      auto &vec = delayStageNames[d.getOperation()];
      for (unsigned i=0;i<vec.size();++i) {
        os << "var " << vec[i] << " : ";
        (void)emitType(d.getLoc(), d.getType());
        os << ";\n";
      }
    });
  }

  void initDelaysIfNeeded(spechls::KernelOp kernel) {
    if (!opts.declareDelaysAsState) return;
    kernel->walk([&](spechls::DelayOp d){
      // If an explicit init is present and not self-initialized, set s{depth-1}
      // else default all stages to 0.
      auto &vec = delayStageNames[d.getOperation()];
      if (d.getInit()) {
        // Initialize the tail (oldest) stage with provided init; others to 0.
        for (unsigned i=0;i<vec.size()-1;++i) {
          os << vec[i] << " = 0;\n"; // default zero
        }
        os << vec.back() << " = "; (void)emitOperand(d.getInit()); os << ";\n";
      } else {
        for (auto &n : vec) os << n << " = 0;\n";
      }
    });
  }

  //---- High-level printers ---------------------------------------------------

  LogicalResult print(Operation &op);

  LogicalResult emitKernel(spechls::KernelOp kernel) {
    // Pre-scan to collect gamma/mux signatures for preamble.
    kernel->walk([&](Operation *o){
      if (auto g = dyn_cast<spechls::GammaOp>(o)) {
        unsigned selBW = 0, elemBW = 0; bool isBool=false;
        if (auto st = dyn_cast<IntegerType>(g.getSelect().getType())) selBW = st.getWidth();
        if (auto it = dyn_cast<IntegerType>(g.getResult().getType())) { elemBW = it.getWidth(); isBool = elemBW==1; }
        (void)ensureGammaDefine(g.getInputs().size(), isBool?1:elemBW, selBW, isBool);
      } else if (auto m = dyn_cast<circt::comb::MuxOp>(o)) {
        unsigned elemBW = 0; bool isBool=false;
        if (auto it = dyn_cast<IntegerType>(m.getTrueValue().getType())) { elemBW = it.getWidth(); isBool = elemBW==1; }
        (void)ensureGammaDefine(/*arity*/2, isBool?1:elemBW, /*selBW*/0, isBool);
      }
    });

    // module header
    os << "module " << sanitize(kernel.getName()) << " {\n"; os.indent();

    // emit preamble (`define` helpers)
    if (!preamble.empty()) os << preamble << "\n";

    // 1) inputs (kernel region arguments)
    for (BlockArgument arg : kernel.getArguments()) {
      os << "input " ; (void)emitType(kernel.getLoc(), arg.getType()); os << " " << getOrCreate(arg) << ";\n";
    }

    // 2) state vars (mus)
    kernel->walk([&](spechls::MuOp mu){
      // Bind a stable name to the mu result (used as the state var name)
      (void)getOrCreate(mu.getResult());
      os << "var " << getOrCreate(mu.getResult()) << " : "; (void)emitType(mu.getLoc(), mu.getType()); os << ";\n";
    });

    // 3) optional: delays as state
    declareDelayVarsIfNeeded(kernel);

    // 4) init { ... }
    os << "init {\n"; os.indent();
    // mu init
    kernel->walk([&](spechls::MuOp mu){
      os << getOrCreate(mu.getResult()) << " = "; (void)emitOperand(mu.getInitValue()); os << ";\n";
    });
    // delay init
    initDelaysIfNeeded(kernel);
    os.unindent(); os << "}\n";

    // 5) procedure step() returns (...)
    // returns = (mu_n for all mu, delay_in for each DelayOp if enabled)
    SmallVector<std::pair<std::string, Type>> retNames;
    kernel->walk([&](spechls::MuOp mu){
      retNames.emplace_back((getOrCreate(mu.getResult()).str() + "_n"), mu.getType());
    });
    if (opts.declareDelaysAsState) {
      kernel->walk([&](spechls::DelayOp d){
        // create a local name to hold the computed new head value
        std::string nm = sanitize(getOrCreate(d.getResult()).str()) + "_in";
        delayStepInputName[d.getOperation()] = nm;
        retNames.emplace_back(nm, d.getType());
      });
    }

    // signature
    os << "procedure step() returns (";
    for (size_t i=0;i<retNames.size();++i) {
      if (i) os << ", ";
      os << retNames[i].first << " : "; (void)emitType(kernel.getLoc(), retNames[i].second);
    }
    os << ") {\n"; os.indent();

    // Body: combinational evaluation in topological order
    {
      Scope bodyScope(*this);
      Block *body = &kernel.getBody().front();
      mlir::sortTopologically(body, spechls::topologicalSortCriterion);
      for (Operation &op : *body) {
        if (isa<spechls::ExitOp>(op)) continue; // no loop to break in Uclid
        if (failed(print(op))) return failure();
      }

      // Bind mu_n returns from loop values
      kernel->walk([&](spechls::MuOp mu){
        os << (getOrCreate(mu.getResult()).str() + "_n") << " = ";
        (void)emitOperand(mu.getLoopValue());
        os << ";\n";
      });

      // For delays: capture the newly computed input value
      if (opts.declareDelaysAsState) {
        kernel->walk([&](spechls::DelayOp d){
          os << delayStepInputName[d.getOperation()] << " = ";
          if (d.getEnable()) {
            // if enable then input else current s0 (hold)
            os << "ite("; (void)emitOperand(d.getEnable()); os << ", ";
            (void)emitOperand(d.getInput()); os << ", " << getDelayStageName(d, 0) << ")";
          } else {
            (void)emitOperand(d.getInput());
          }
          os << ";\n";
        });
      }

      // return tuple
      os << "return (";
      for (size_t i=0;i<retNames.size();++i) { if (i) os << ", "; os << retNames[i].first; }
      os << ");\n";
    }

    os.unindent(); os << "}\n"; // end procedure

    // 6) next { call (state') = step();  delay shifts; }
    os << "next {\n"; os.indent();

    // Build LHS list for call: mu' and delay_s0'
    os << "call (";
    bool first = true;
    kernel->walk([&](spechls::MuOp mu){ if (!first) os << ", "; first=false; os << getOrCreate(mu.getResult()) << "'"; });
    if (opts.declareDelaysAsState) {
      kernel->walk([&](spechls::DelayOp d){ os << ", " << getDelayStageName(d, 0) << "'"; });
    }
    os << ") = step();\n";

    // Delay shifts: s{i}' = s{i-1}; already s0' set by call
    if (opts.declareDelaysAsState) {
      kernel->walk([&](spechls::DelayOp d){
        auto &vec = delayStageNames[d.getOperation()];
        for (unsigned i=1;i<vec.size();++i) {
          os << vec[i] << "' = " << vec[i-1] << ";\n";
        }
      });
    }

    os.unindent(); os << "}\n"; // end next

    os.unindent(); os << "}\n"; // end module
    return success();
  }
};

//------------------------------------------------------------------------------
// Operation printers
//------------------------------------------------------------------------------

static StringRef bvCmpToken(circt::comb::ICmpPredicate p, bool &isUnsigned) {
  switch (p) {
  case circt::comb::ICmpPredicate::eq: return "==";
  case circt::comb::ICmpPredicate::ne: return "!=";
  case circt::comb::ICmpPredicate::slt: isUnsigned=false; return "<";
  case circt::comb::ICmpPredicate::sle: isUnsigned=false; return "<=";
  case circt::comb::ICmpPredicate::sgt: isUnsigned=false; return ">";
  case circt::comb::ICmpPredicate::sge: isUnsigned=false; return ">=";
  case circt::comb::ICmpPredicate::ult: isUnsigned=true;  return "<_u";
  case circt::comb::ICmpPredicate::ule: isUnsigned=true;  return "<=_u";
  case circt::comb::ICmpPredicate::ugt: isUnsigned=true;  return ">_u";
  case circt::comb::ICmpPredicate::uge: isUnsigned=true;  return ">=_u";
  }
  return "==";
}

LogicalResult UclidEmitter::print(Operation &op) {
  return llvm::TypeSwitch<Operation*, LogicalResult>(&op)
    // Module is handled at entry point
    .Case<ModuleOp>([&](auto){ return success(); })

    // Kernel (top-level emission)
    .Case<spechls::KernelOp>([&](auto k){ return emitKernel(k); })

    // --- Comb ops ---
    .Case<circt::comb::AddOp>([&](auto o){ return printBinary(o.getOperation(), "+"); })
    .Case<circt::comb::SubOp>([&](auto o){ return printBinary(o.getOperation(), "-"); })
    .Case<circt::comb::MulOp>([&](auto o){ return printBinary(o.getOperation(), "*"); })
    .Case<circt::comb::AndOp>([&](auto o){ return printBinary(o.getOperation(), "&"); })
    .Case<circt::comb::OrOp >( [&](auto o){ return printBinary(o.getOperation(), "|"); })
    .Case<circt::comb::XorOp>([&](auto o){ return printBinary(o.getOperation(), "^"); })
    .Case<circt::comb::ShlOp>([&](auto o){ return printBinary(o.getOperation(), "<<"); })
    .Case<circt::comb::ShrUOp>([&](auto o){ return printBinary(o.getOperation(), ">>_u"); })
    .Case<circt::comb::ShrSOp>([&](auto o){ return printBinary(o.getOperation(), ">>"); })
    .Case<circt::comb::DivUOp>([&](auto o){ return printBinary(o.getOperation(), "/_u"); })
    .Case<circt::comb::DivSOp>([&](auto o){ return printBinary(o.getOperation(), "/"); })
    .Case<circt::comb::ModUOp>([&](auto o){ return printBinary(o.getOperation(), "%_u"); })
    .Case<circt::comb::ModSOp>([&](auto o){ return printBinary(o.getOperation(), "%"); })
    .Case<circt::comb::ConcatOp>([&](auto o){
      Operation *op = o.getOperation();
      os << getOrCreate(op->getResult(0)) << " = ";
      // a ++ b ++ c
      for (unsigned i=0;i<op->getNumOperands();++i) {
        if (i) os << " ++ ";
        if (failed(emitOperand(op->getOperand(i)))) return failure();
      }
      os << ";\n"; return success();
    })
    .Case<circt::comb::ExtractOp>([&](auto o){
      Operation *op = o.getOperation();
      // Uclid supports bv slice with [high:low]
      unsigned low = o.getLowBit();
      unsigned width = o.getType().getIntOrFloatBitWidth();
      unsigned high = low + width - 1;
      os << getOrCreate(op->getResult(0)) << " = ";
      if (failed(emitOperand(op->getOperand(0)))) return failure();
      os << "[" << high << ":" << low << "]" << ";\n";
      return success();
    })
    .Case<circt::comb::ICmpOp>([&](auto o){
      Operation *op = o.getOperation();
      bool isUnsigned=false; StringRef tok = bvCmpToken(o.getPredicate(), isUnsigned);
      os << getOrCreate(op->getResult(0)) << " = ";
      if (failed(emitOperand(o.getLhs()))) return failure();
      os << " " << tok << " ";
      if (failed(emitOperand(o.getRhs()))) return failure();
      os << ";\n"; return success();
    })
    .Case<circt::comb::MuxOp>([&](auto o){
      unsigned elemBW = 0; bool isBool=false;
      if (auto it = dyn_cast<IntegerType>(o.getTrueValue().getType())) { elemBW = it.getWidth(); isBool = elemBW==1; }
      std::string fname = ensureGammaDefine(/*arity*/2, isBool?1:elemBW, /*selBW*/0, isBool);
      os << getOrCreate(o.getResult()) << " = " << fname << "(";
      if (failed(emitOperand(o.getCond()))) return failure();
      os << ", "; if (failed(emitOperand(o.getTrueValue()))) return failure();
      os << ", "; if (failed(emitOperand(o.getFalseValue()))) return failure();
      os << ");\n"; return success();
    })

    // --- HW ops ---
    .Case<circt::hw::BitcastOp>([&](auto o){
      // Best-effort: cast is no-op on Uclid side if widths align; otherwise emit explicit type cast
      os << getOrCreate(o.getResult()) << " = ("; (void)emitType(o.getLoc(), o.getType()); os << ") ";
      if (failed(emitOperand(o.getInput()))) return failure();
      os << ";\n"; return success();
    })

    // --- SpecHLS ops ---
    .Case<spechls::GammaOp>([&](auto g){
      unsigned selBW = 0, elemBW = 0; bool isBool=false;
      if (auto st = dyn_cast<IntegerType>(g.getSelect().getType())) selBW = st.getWidth();
      if (auto it = dyn_cast<IntegerType>(g.getResult().getType())) { elemBW = it.getWidth(); isBool = elemBW==1; }
      const unsigned arity = g.getInputs().size();
      std::string fname = ensureGammaDefine(arity, isBool?1:elemBW, selBW, isBool);
      os << "{ // gamma via " << fname << "\n"; os.indent();
      os << getOrCreate(g.getResult()) << " = " << fname << "(";
      if (failed(emitOperand(g.getSelect()))) return failure();
      for (Value a : g.getInputs()) { os << ", "; if (failed(emitOperand(a))) return failure(); }
      os << ");\n";
      os.unindent(); os << "}\n"; return success();
    })
    .Case<spechls::LUTOp>([&](auto lut){
      // Inline case for small LUTs
      auto table = lut.getContents();
      unsigned outBW = lut.getType().getIntOrFloatBitWidth(); (void)outBW;
      os << getOrCreate(lut.getResult()) << " = case "; if (failed(emitOperand(lut.getIndex()))) return failure(); os << " of\n";
      unsigned idx=0; for (auto v : table) { os << "  " << idx++ << ": " << v << ";\n"; }
      os << "  otherwise: 0;\n";
      os << "esac;\n";
      return success();
    })
    .Case<spechls::DelayOp>([&](auto d){
      // Reading a delay returns the oldest stage; in step we reference the last stage var.
      if (!opts.declareDelaysAsState) {
        d.emitOpError("delay modeling disabled in Uclid emitter");
        return failure();
      }
      auto &vec = delayStageNames[d.getOperation()]; if (vec.empty()) (void)getDelayStageName(d, 0);
      os << getOrCreate(d.getResult()) << " = " << vec.back() << ";\n"; // head == oldest
      return success();
    })
    .Case<spechls::MuOp>([&](auto){
      // No direct emission here: reads of mu use its bound name; mu_n assignment is in step epilogue.
      return success();
    })
    .Case<spechls::ExitOp>([&](auto){ return success(); })

    .Default([&](Operation *u) {
      auto message = "no Uclid printer yet for this operation type" + std::string(u->getName().getStringRef());
      u->emitOpError(message);
      return failure();
    });
}

//------------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------------

LogicalResult translateToUclid(Operation *op, raw_ostream &os) {
  TranslationToUclidOptions options = {};
  raw_indented_ostream ios(os);
  UclidEmitter emitter(ios, std::move(options));
  return emitter.print(*op);
  return success();
}


  LogicalResult dumpToUclid(Operation *op, llvm::StringRef path) {
  std::string err;
  auto out = mlir::openOutputFile(path, &err); // returns ToolOutputFile
  if (!out) {
    llvm::errs() << err << "\n";
    return mlir::failure();
  }
  LogicalResult res = spechls::translateToUclid(op, out->os());
  if (succeeded(res)) out->keep();            // persist the file
  return res;
}

} // namespace spechls
