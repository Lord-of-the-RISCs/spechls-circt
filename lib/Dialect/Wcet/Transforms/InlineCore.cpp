//===- InlineTasks.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

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

#define DEBUG_TYPE "inlineCore"

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_INLINECOREPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

struct MyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned, IRMapping &valueMapping) const final {
    return true;
  }
};

namespace wcet {

struct InlineCorePass : public impl::InlineCorePassBase<InlineCorePass> {

  using InlineCorePassBase::InlineCorePassBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    // Add inliner interface to the comb dialect if not registered
    auto *combDialect = ctx->getLoadedDialect("comb");
    assert(combDialect);
    if (!combDialect->getRegisteredInterface(MyInlinerInterface::getInterfaceID()))
      combDialect->addInterface<MyInlinerInterface>();

    // Add inliner interface to the spechls dialect if not registered
    auto *spechlsDialect = ctx->getLoadedDialect("spechls");
    assert(spechlsDialect);
    if (!spechlsDialect->getRegisteredInterface(MyInlinerInterface::getInterfaceID()))
      spechlsDialect->addInterface<MyInlinerInterface>();

    OpPassManager pm = PassManager::on<mlir::ModuleOp>(ctx);
    pm.addPass(mlir::createInlinerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace wcet
