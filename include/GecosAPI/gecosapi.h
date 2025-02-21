#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOpsDialect.cpp.inc"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif
void initSpecHLS();
MlirModule parseMLIR(const char *mlir);
void traverseRegion(MlirRegion region);
void traverseMLIR(MlirModule module);
void pass(const char *mlir);
#ifdef __cplusplus
}
#endif

/* Autogenerated by mlir-tblgen; don't manually edit. */

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registration for the entire group
MLIR_CAPI_EXPORTED void createSchedulePass(void);
MLIR_CAPI_EXPORTED void registerSchedulePass(void);

// Registration for the entire group
MLIR_CAPI_EXPORTED void createMobilityPass(void);
MLIR_CAPI_EXPORTED void registerMobilityPass(void);
MLIR_CAPI_EXPORTED void createConfigurationExcluderPass(void);
MLIR_CAPI_EXPORTED void registerConfigurationExcluderPass(void);

#ifdef __cplusplus
}
#endif
