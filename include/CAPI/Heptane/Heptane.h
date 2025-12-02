#ifndef HEPTANE_H
#define HEPTANE_H

#include <mlir-c/IR.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Support/LLVM.h>

extern "C" {

MlirModule parseMLIR(const char *str);

void destroyMLIR(MlirModule module);

void mlirDumpModule(MlirModule module);
size_t mlirWcetAnalysis(MlirModule module, mlir::SmallVector<size_t> &instrs);
}

#endif // !HEPTANE_H
