//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU Pragma: keep
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/RegionUtils.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/Passes.h>

#include <mlir/IR/UseDefLists.h>
#include <utility>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOPOSORTPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

class SpecHLSTopoSortPass : public spechls::impl::SpecHLSTopoSortPassBase<SpecHLSTopoSortPass> {
public:
  using SpecHLSTopoSortPassBase::SpecHLSTopoSortPassBase;

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](spechls::TaskOp task) {
      mlir::sortTopologically(&task.getBody().front(), spechls::topologicalSortCriterion);
    });
    mod.walk([&](spechls::KernelOp kernel) {
      mlir::sortTopologically(&kernel.getBody().front(), spechls::topologicalSortCriterion);
    });
  }
};

} // namespace