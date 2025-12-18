//===- EnsureClockingPorts.h ------------------------------------*- C++ -*-===//
//
// Utilities to ensure and wire (clk, ce, rst) input ports on hw.module ops,
// and propagate them recursively through parent modules + instances.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_UTILS_ENSURECLOCKINGPORTS_H
#define CIRCT_UTILS_ENSURECLOCKINGPORTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h" // mlir::FailureOr

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

namespace circt::hw {

/// Configuration for the 3 clocking ports we want to guarantee everywhere.
struct ClockingPortsOptions {
  /// Port names to ensure (inputs).
  llvm::StringRef clkName = "clk";
  llvm::StringRef ceName  = "ce";
  llvm::StringRef rstName = "rst";

  /// Types to use when a port must be created.
  /// - If clkType is null, implementations typically default to hw::ClockType.
  /// - If ceType/rstType are null, implementations typically default to i1.
  mlir::Type clkType;
  mlir::Type ceType;
  mlir::Type rstType;

  /// If true, new ports are appended at the end of the input list.
  bool appendIfMissing = true;

  /// If true and a port already exists, still (re)wire instance operands to match.
  bool rewireIfPresent = true;
};

/// Result of "ensure port": whether it existed or was created, and how to address it.
struct EnsuredInputPort {
  mlir::StringAttr name;   ///< Port name attr.
  mlir::Type type;         ///< Final port type.
  unsigned inputIndex = 0; ///< Index in the module's *input* port list.
  mlir::Value value;       ///< Block argument value for hw.module bodies; null for extern/generated.
  bool created = false;    ///< True if the port was created by the helper.
};

struct EnsuredClockingPorts {
  EnsuredInputPort clk;
  EnsuredInputPort ce;
  EnsuredInputPort rst;
};

/// Return the "root" symbol table op used to resolve hw.module + instances
/// (typically the surrounding mlir::ModuleOp).
mlir::ModuleOp getEnclosingSymbolTableModule(mlir::Operation *anchor);

/// Ensure a single *input* port exists on `mod` (create it if missing).
/// Returns the ensured port descriptor (including final input index).
mlir::FailureOr<EnsuredInputPort>
ensureInputPort(hw::HWModuleOp mod, llvm::StringRef portName, mlir::Type portType,
                bool appendIfMissing = true);

/// Ensure (clk, ce, rst) exist as *input* ports on `mod`.
mlir::FailureOr<EnsuredClockingPorts>
ensureClockingPorts(hw::HWModuleOp mod, const ClockingPortsOptions &opts = {});

/// Collect all hw.instance ops in `root` that instantiate `targetModuleName`.
/// (Implementation can scan IR or rely on an instance graph if available.)
void collectInstancesOf(mlir::ModuleOp root, mlir::StringAttr targetModuleName,
                        llvm::SmallVectorImpl<hw::InstanceOp> &out);

/// Resolve the referenced HWModuleOp of an instance (if it is a hw.module).
mlir::FailureOr<hw::HWModuleOp> resolveCalleeModule(hw::InstanceOp inst);

/// Ensure the instance's operand list matches the callee input port count, and
/// set the operand at `inputIndex` to `newValue`. Also updates naming metadata
/// if needed (e.g. argNames) to keep IR consistent.
mlir::LogicalResult
setOrInsertInstanceInput(hw::InstanceOp inst, unsigned inputIndex, mlir::Value newValue,
                         mlir::StringAttr inputName = mlir::StringAttr{});

/// Wire parent's (clk, ce, rst) values into a specific instance that calls `child`,
/// matching by the ensured input indices.
mlir::LogicalResult
wireClockingPortsIntoInstance(hw::InstanceOp inst,
                              const EnsuredClockingPorts &parentPorts,
                              const EnsuredClockingPorts &childPorts,
                              bool rewireIfPresent = true);

/// Top-level helper (your requested entry point):
/// - takes a HWModuleOp (leaf)
/// - ensures clk/ce/rst on it
/// - finds all parents instantiating it, ensures same ports recursively
/// - wires parent->child instance operands for clk/ce/rst at each level
mlir::LogicalResult
ensureAndPropagateClockingPorts(hw::HWModuleOp leaf,
                                const ClockingPortsOptions &opts = {});

} // namespace circt::hw

#endif // CIRCT_UTILS_ENSURECLOCKINGPORTS_H
