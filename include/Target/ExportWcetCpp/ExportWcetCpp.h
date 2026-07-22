//===- ExportWcetCpp.h - wcet.core -> C++ CoreAnalysis -------------------===//
//
//
//
//===---------------------------------------------------------------------===//

#ifndef SPECHLS_TARGET_EXPORTWCET_CPP_H
#define SPECHLS_TARGET_EXPORTWCET_CPP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_os_ostream.h"

namespace spechls {

/// Translate every wcet.core op found in \p module into self-contained
/// C++ file that implements the CoreAnalysis interface, written to \p os.
mlir::LogicalResult exportWcetCpp(mlir::ModuleOp module, llvm::raw_os_ostream &os);

///  Register the "export-wcet-cpp" TranslateFromMlir command-line option.
///  Call this before mlir::MlirTranslateMain().
void registerExportWcetCpp();

} // namespace spechls

#endif // !SPECHLS_TARGET_EXPORTWCET_CPP_H
