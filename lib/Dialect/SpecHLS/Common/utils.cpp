//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"


#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU pragma: keep

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"


#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU pragma: keep

using namespace mlir;
using namespace circt;
#define debug llvm::outs
//#define debug llvm::nulls


/// Helper to walk back array SSA chain to the defining MuOp.
static spechls::MuOp findRootMu(Value arrayVal) {
  while (arrayVal!=nullptr) {
    debug() << "Traversing " << arrayVal << "\n";
    if (auto mu = arrayVal.getDefiningOp<spechls::MuOp>()) {
      debug() << "Found MuOp: " << mu << "\n";
      return mu;
    }
    if (auto alpha = arrayVal.getDefiningOp<spechls::AlphaOp>()) {

      arrayVal = alpha.getArray();
      continue;
    }
    if (auto mux = arrayVal.getDefiningOp<comb::MuxOp>()) {

      arrayVal = mux.getOperand(0);
      continue;
    }
    if (auto sync = arrayVal.getDefiningOp<spechls::SyncOp>()) {

      arrayVal = sync.getOperand(0);
      continue;
    }


    return nullptr;
  }
}
