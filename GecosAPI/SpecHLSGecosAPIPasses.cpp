//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/ScheduleDialect.h"
#include "Dialect/Schedule/ScheduleOps.h"

#include "mlir-c/Conversion.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/IR.h>

#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/FSM.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>

#include <CAPI/SpecHLS.h>

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir-c/Conversion.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Transforms.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/IR.h>

#include "mlir/Transforms/Transforms.capi.h.inc"
#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/FSM.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>

#include <CAPI/SpecHLS.h>

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir-c/Pass.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
// #include "mlir/CAPI/Support.h"
// #include "mlir/CAPI/Utils.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

extern "C" {

// FIXME move into include file
MlirPass mlirCreateSchedulePass(void);
MlirPass mlirCreateMobilityPass(void);
MlirPass mlirCreateLocalMobilityPass(void);
MlirPass mlirCreateConfigurationExcluderPass(void);
MlirPass mlirCreateExportVitisHLS(void);

#define DEFINE_GECOS_API_PASS(name, pass)                                      \
                                                                               \
  MlirPass mlirCreate##pass();                                                 \
                                                                               \
  MlirModule name(MlirModule module) {                                         \
    MlirContext ctx = mlirModuleGetContext(module);                            \
    MlirOperation op = mlirModuleGetOperation(module);                         \
    MlirPassManager pm = wrap(                                                 \
        new mlir::PassManager(unwrap(ctx), mlir::ModuleOp::getOperationName(), \
                              mlir::PassManager::Nesting::Implicit));          \
    MlirPass p = mlirCreate##pass();                                           \
    mlirPassManagerAddOwnedPass(pm, p);                                        \
    MlirLogicalResult success = mlirPassManagerRunOnOp(pm, op);                \
    if (mlirLogicalResultIsFailure(success)) {                                 \
      fprintf(stderr, "Unexpected failure running pass manager.\n");           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    mlirPassManagerDestroy(pm);                                                \
    return module;                                                             \
  }

DEFINE_GECOS_API_PASS(scheduleMLIR, SchedulePass)
DEFINE_GECOS_API_PASS(canonicalizeMLIR, TransformsCanonicalizer)
DEFINE_GECOS_API_PASS(mobilityMLIR, MobilityPass)

DEFINE_GECOS_API_PASS(configurationExcluderMLIR, ConfigurationExcluderPass)

DEFINE_GECOS_API_PASS(exportVitisHLS, ExportVitisHLS)
DEFINE_GECOS_API_PASS(yosysOptimizer, YosysOptimizerPass)
DEFINE_GECOS_API_PASS(groupControl, GroupControlNodePass)
DEFINE_GECOS_API_PASS(factorGammaInputs, FactorGammaInputsPass)
DEFINE_GECOS_API_PASS(mergeLUTs, MergeLookUpTablesPass)
DEFINE_GECOS_API_PASS(mergeGammas, MergeGammasPass)
DEFINE_GECOS_API_PASS(eliminateRedundantGammaInputs,
                      EliminateRedundantGammaInputsPass)
DEFINE_GECOS_API_PASS(inlineModule, InlineModulesPass)
}
