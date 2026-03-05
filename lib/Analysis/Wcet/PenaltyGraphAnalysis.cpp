//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "Analysis/Wcet/PenaltyGraphAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "Support/WcetUtils.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;
namespace llvm {
template <>
struct DenseMapInfo<wcet::state> {
  static inline wcet::state getEmptyKey() { return wcet::StateStruct(-1, 0, {}); }

  static inline wcet::state getTombstoneKey() { return wcet::StateStruct(-2, 0, {}); }

  static unsigned getHashValue(const wcet::state &key) {
    return (hash_value(key.layers)) ^ (hash_value(key.pen) << 1) ^ (hash_value(key.st.size()) << 2);
  }

  static bool isEqual(const wcet::state &lhs, const wcet::state &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace wcet {

PenaltyGraphAnalysis::PenaltyGraphAnalysis(ModuleOp mod, SmallVector<size_t> instrs) {
  wcet = 0;
  wcet::CoreOp coreAnalyzed = nullptr;
  mod->walk([&coreAnalyzed](wcet::CoreOp c) {
    if (c->hasAttr("wcet.cpuCore"))
      coreAnalyzed = c;
  });
  if (!coreAnalyzed)
    return;

  std::string instrStr = "instrs=";
  for (auto ins : instrs) {
    instrStr += std::to_string(ins) + ",";
  }
  instrStr[instrStr.size() - 1] = ' ';
  llvm::errs() << instrStr << "\n";

  mlir::PassManager pm = mlir::PassManager::on<mlir::ModuleOp>(mod->getContext());
  auto unrollPass = wcet::createUnrollInstrPass();
  if (failed(unrollPass->initializeOptions(instrStr, [](const Twine &msg) {
        llvm::errs() << msg << "\n";
        return failure();
      })))
    return;
  pm.addPass(std::move(unrollPass));
  pm.addPass(wcet::createInlineCorePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(wcet::createLongestPathPass());
  if (failed(pm.run(mod))) {
    return;
  }

  wcet::DummyOp lastDummy = nullptr;
  mod->walk([&lastDummy](wcet::DummyOp d) {
    if (d->hasAttr("wcet.next"))
      lastDummy = d;
  });

  mlir::IntegerAttr attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(lastDummy->getAttr("wcet.WCET"));
  wcet = attr.getInt();

  lastDummy->getParentOfType<wcet::CoreOp>()->erase();
}

} // namespace wcet
