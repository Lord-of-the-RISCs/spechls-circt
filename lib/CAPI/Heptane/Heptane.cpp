//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "CAPI/Dialect/Schedule.h"
#include "CAPI/Dialect/SpecHLS.h"
#include "CAPI/Dialect/Wcet.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/Support/Format.h>

#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/Debug.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/LLHD.h>
#include <circt-c/Dialect/LTL.h>
#include <circt-c/Dialect/Moore.h>
#include <circt-c/Dialect/SSP.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>
#include <circt-c/Dialect/Sim.h>
#include <circt-c/Dialect/Synth.h>
#include <circt-c/Dialect/Verif.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <circt/Dialect/SSP/SSPOps.h>
#include <ios>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "CAPI/Heptane/Heptane.h"

using namespace mlir;

namespace {

wcet::CoreOp createAnalyseCore(IRRewriter &rewriter, ModuleOp &top, wcet::CoreOp &analyzedCore,
                               SmallVector<std::optional<IntegerAttr>> &state, SmallVector<Type> &types,
                               size_t instrs) {

  rewriter.setInsertionPointToEnd(top.getBody());
  wcet::CoreOp result = rewriter.create<wcet::CoreOp>(rewriter.getUnknownLoc(), rewriter.getFunctionType({}, {}),
                                                      rewriter.getStringAttr(CORE_ANALYSIS_NAME));
  result->setAttr("wcet.analysis", rewriter.getUnitAttr());
  rewriter.setInsertionPointToEnd(&result.getBody().front());

  SmallVector<Value> coreInputs;
  auto instr =
      rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), analyzedCore.getArgumentTypes().front(), instrs);
  coreInputs.push_back(instr);

  SmallVector<Value> dummyInputs;
  for (auto st : llvm::enumerate(state)) {
    if (st.value().has_value()) {
      dummyInputs.push_back(
          rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), types[st.index()], st.value().value()));
    } else {
      dummyInputs.push_back(rewriter.create<wcet::InitOp>(rewriter.getUnknownLoc(), types[st.index()], "Unknown"));
    }
  }

  auto firstDummy = rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), types, dummyInputs);
  firstDummy->setAttr("wcet.current", rewriter.getUnitAttr());
  for (auto out : firstDummy.getOutputs()) {
    coreInputs.push_back(out);
  }

  auto coreInstance = rewriter.create<wcet::CoreInstanceOp>(rewriter.getUnknownLoc(), analyzedCore, coreInputs);

  auto secondDummy = rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), coreInstance->getResultTypes(),
                                                    coreInstance->getResults());
  secondDummy->setAttr("wcet.next", rewriter.getUnitAttr());

  rewriter.create<wcet::CommitOp>(rewriter.getUnknownLoc(), result->getResultTypes(), ValueRange());
  // result->dumpPretty();
  return result;
}

} // namespace
extern "C" {

//===--------------------------------------------------------------------------------------------------------------===//
// Utility functions
//===--------------------------------------------------------------------------------------------------------------===//

// const char *getCStringDataFromMlirStringRef(MlirStringRef str) { return str.data; }
//
// size_t getCStringSizeFromMlirStringRef(MlirStringRef str) { return str.length; }
//
// const char *getCStringDataFromMlirIdentifier(MlirIdentifier identifier) {
//   MlirStringRef str = mlirIdentifierStr(identifier);
//   return str.data;
// }
//
// size_t getCStringSizeFromMlirIdentifier(MlirIdentifier identifier) {
//   MlirStringRef str = mlirIdentifierStr(identifier);
//   return str.length;
// }
//
// char getCharAt(const char *v, int offset) { return v[offset]; }
//
// MlirIdentifier mlirOperationGetAttributeNameAt(MlirOperation op, int64_t pos) {
//   return mlirOperationGetAttribute(op, pos).name;
// }
//
// MlirAttribute mlirOperationGetAttributeAt(MlirOperation op, int64_t pos) {
//   return mlirOperationGetAttribute(op, pos).attribute;
// }

//===--------------------------------------------------------------------------------------------------------------===//
// Passes
//===--------------------------------------------------------------------------------------------------------------===//

MlirModule parseMLIR(const char *str) {
  MlirContext context = mlirContextCreate();

  // Register required dialects.
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hw__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__comb__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ssp__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__spechls__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__schedule__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__wcet__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__seq__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__synth__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sv__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llhd__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__cf__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__math__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sim__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__verif__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__moore__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ltl__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__debug__(), context);

  MlirStringRef wrapped = mlirStringRefCreateFromCString(str);
  return mlirModuleCreateParse(context, wrapped);
}

void destroyMLIR(MlirModule module) {
  MlirContext context = mlirModuleGetContext(module);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
}

#define DEFINE_WCET_API_PASS(func, namespace, name)                                                                    \
  MlirModule func(MlirModule mod) {                                                                                    \
    mlir::ModuleOp module = unwrap(mod);                                                                               \
    mlir::PassManager pm(module.getContext(), mlir::ModuleOp::getOperationName(),                                      \
                         mlir::PassManager::Nesting::Implicit);                                                        \
    pm.addPass(namespace ::create##name());                                                                            \
    if (mlir::failed(pm.run(module)))                                                                                  \
      llvm::errs() << "Unexpected failure running pass manager.\n";                                                    \
    return wrap(module);                                                                                               \
  }

void mlirDumpModule(MlirModule module) { unwrap(module)->dump(); }

size_t mlirWcetAnalysis(MlirModule module, mlir::SmallVector<size_t> &instrs) {
  //==== Setup Analysis
  size_t wcet = 0;
  auto mod = unwrap(module);
  mlir::IRRewriter rewriter(mod->getContext());

  wcet::CoreOp analyzedCore = nullptr;
  mod->walk([&](wcet::CoreOp c) {
    if (c->hasAttr("wcet.cpuCore"))
      analyzedCore = c;
  });

  mlir::SmallVector<std::optional<mlir::IntegerAttr>> state;
  mlir::SmallVector<Type> stateTypes;

  for (auto type : analyzedCore.getResultTypes()) {
    stateTypes.push_back(type);
    state.push_back({});
  }

  // llvm::errs() << "begins analysis\n";
  for (auto instr : instrs) {
    auto core = createAnalyseCore(rewriter, mod, analyzedCore, state, stateTypes, instr);
    auto pm = mlir::PassManager::on<mlir::ModuleOp>(mod->getContext());

    pm.addPass(wcet::createInlineCorePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(wcet::createConstantPropPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(wcet::createLongestPathPass());
    if (failed(pm.run(mod))) {
      llvm::errs() << "pipeline failed\n";
      return 0;
    }

    size_t currentWcet = 0;
    wcet::DummyOp lastDum = nullptr;
    for (auto d : core.getOps<wcet::DummyOp>()) {
      if (!d->hasAttr("wcet.next"))
        continue;
      auto wAttr = dyn_cast_or_null<IntegerAttr>(d->getAttr("wcet.dists"));
      if (wAttr) {
        currentWcet = wAttr.getInt();
        lastDum = d;
        break;
      }
    }
    if (!lastDum) {
      llvm::errs() << "instruction fail " << instr << "\n";
      return 0;
    }
    // llvm::errs() << "wcet of the instr: " << llvm::format("0x%08x", instr) << " - " << currentWcet << "\n";
    // core->dumpPretty();

    assert(lastDum->getResultTypes().size() == analyzedCore.getResultTypes().size());
    SmallVector<std::optional<IntegerAttr>> dumResult;
    for (auto d : lastDum.getInputs()) {
      auto lastResultOp = dyn_cast_or_null<circt::hw::ConstantOp>(d.getDefiningOp());
      if (lastResultOp) {
        dumResult.push_back(lastResultOp.getValueAttr());
        continue;
      }
      dumResult.push_back({});
    }

    state.clear();
    assert(analyzedCore.getResultTypes().size() + 1 == analyzedCore.getArgumentTypes().size());
    for (size_t i = 0; i < analyzedCore.getResultTypes().size(); i++) {
      auto nbPred = dyn_cast_or_null<IntegerAttr>(analyzedCore.getArgAttr(i + 1, "wcet.nbPred"));
      if (!nbPred || nbPred.getInt() == 0) {
        state.push_back(dumResult[i]);
      } else if (nbPred.getInt() > (int64_t)currentWcet) {
        state.push_back(dumResult[i - currentWcet]);
      } else {
        IntegerType it = dyn_cast_or_null<IntegerType>(stateTypes[i]);
        if (it && it.getWidth() == 1) {
          state.push_back(rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
        } else {
          state.push_back({});
        }
      }
    }
    core->erase();
    wcet += currentWcet;
  }
  return wcet;
}
}
