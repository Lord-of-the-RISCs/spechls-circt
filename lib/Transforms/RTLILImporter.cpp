#include "RTLILImporter.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "kernel/rtlil.h"                  // from @at_clifford_yosys
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"     // from @llvm-project
#include "llvm/ADT/MapVector.h"            // from @llvm-project

namespace mlir {

using ::Yosys::RTLIL::Module;
using ::Yosys::RTLIL::SigSpec;
using ::Yosys::RTLIL::Wire;

namespace {

// getTypeForWire gets the MLIR type corresponding to the RTLIL wire. If the
// wire is an integer with multiple bits, then the MLIR type is a tensor of
// bits.
Type getTypeForWire(OpBuilder &b, Wire *wire) {
  auto intTy = b.getI1Type();
  if (wire->width == 1) {
    return intTy;
  }
  return RankedTensorType::get({wire->width}, intTy);
}

} // namespace

llvm::SmallVector<std::string, 10>
getTopologicalOrder(std::stringstream &torderOutput) {
  llvm::SmallVector<std::string, 10> cells;
  std::string line;
  while (std::getline(torderOutput, line)) {
    auto lineCell = line.find("cell ");
    if (lineCell != std::string::npos) {
      cells.push_back(line.substr(lineCell + 5, std::string::npos));
    }
  }
  return cells;
}

void RTLILImporter::addWireValue(Wire *wire, Value value) {
  wireNameToValue.insert(std::make_pair(wire->name.str(), value));
}

Value RTLILImporter::getWireValue(Wire *wire) {
  auto wireName = wire->name.str();
  assert(wireNameToValue.contains(wireName));
  return wireNameToValue.at(wireName);
}

Value RTLILImporter::getBit(
    const SigSpec &conn, ImplicitLocOpBuilder &b,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  // Because the cells are in topological order, and Yosys should have
  // removed redundant wire-wire mappings, the cell's inputs must be a bit
  // of an input wire, in the map of already defined wires (which are
  // bits), or a constant bit.
  assert(conn.is_wire() || conn.is_fully_const() || conn.is_bit());
  if (conn.is_wire()) {
    auto name = conn.as_wire()->name.str();
    assert(wireNameToValue.contains(name));
    return wireNameToValue[name];
  }
  if (conn.is_fully_const()) {
    auto bit = conn.as_const();
    auto constantOp = b.createOrFold<arith::ConstantOp>(
        b.getIntegerAttr(b.getIntegerType(1), bit.as_int()));
    return constantOp;
  }
  // Extract the bit of the multi-bit input or output wire.
  assert(conn.as_bit().is_wire());
  auto bit = conn.as_bit();
  if (retBitValues.contains(bit.wire)) {
    auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
    return retBitValues[bit.wire][offset];
  }
  auto argA = getWireValue(bit.wire);
  auto extractOp = b.create<tensor::ExtractOp>(
      argA, b.create<arith::ConstantIndexOp>(bit.offset).getResult());
  return extractOp;
}

void RTLILImporter::addResultBit(
    const SigSpec &conn, Value result,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  assert(conn.is_wire() || conn.is_bit());
  if (conn.is_wire()) {
    addWireValue(conn.as_wire(), result);
    return;
  }
  // This must be a bit of the multi-bit output wire.
  auto bit = conn.as_bit();
  assert(bit.is_wire() && retBitValues.contains(bit.wire));
  auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
  retBitValues[bit.wire][offset] = result;
}

func::FuncOp
RTLILImporter::importModule(Module *module,
                            const SmallVector<std::string, 10> &cellOrdering) {
  // Gather input and output wires of the module to match up with the block
  // arguments.
  SmallVector<Type, 4> argTypes;
  SmallVector<Wire *, 4> wireArgs;
  SmallVector<Type, 4> retTypes;
  SmallVector<Wire *, 4> wireRet;

  OpBuilder builder(context);
  // Maintain a map from RTLIL output wires to the Values that comprise it
  // in order to reconstruct the multi-bit output.
  llvm::MapVector<Wire *, SmallVector<Value>> retBitValues;
  for (auto *wire : module->wires()) {
    // The RTLIL module may also have intermediate wires that are neither inputs
    // nor outputs.
    if (wire->port_input) {
      argTypes.push_back(getTypeForWire(builder, wire));
      wireArgs.push_back(wire);
    } else if (wire->port_output) {
      retTypes.push_back(getTypeForWire(builder, wire));
      wireRet.push_back(wire);
      retBitValues[wire].resize(wire->width);
    }
  }

  // Build function.
  // TODO(https://github.com/google/heir/issues/111): Pass in data to fix
  // function location.
  FunctionType funcType = builder.getFunctionType(argTypes, retTypes);
  auto function = func::FuncOp::create(
      builder.getUnknownLoc(), module->name.str().replace(0, 1, ""), funcType);
  function.setPrivate();

  auto *block = function.addEntryBlock();
  auto b = ImplicitLocOpBuilder::atBlockBegin(function.getLoc(), block);

  // Map the RTLIL wires to the block arguments' Values.
  for (auto i = 0; i < wireArgs.size(); i++) {
    addWireValue(wireArgs[i], block->getArgument(i));
  }

  // Convert cells to Operations according to topological order.
  for (const auto &cellName : cellOrdering) {
    assert(module->cells_.count("\\" + cellName) != 0 &&
           "expected cell in RTLIL design");
    auto *cell = module->cells_["\\" + cellName];

    SmallVector<Value, 4> inputValues;
    for (const auto &conn : getInputs(cell)) {
      inputValues.push_back(getBit(conn, b, retBitValues));
    }
    auto *op = createOp(cell, inputValues, b);
    auto resultConn = getOutput(cell);
    addResultBit(resultConn, op->getResult(0), retBitValues);
  }

  // Wire up remaining connections.
  for (const auto &conn : module->connections()) {
    auto output = conn.first;
    // These must be output wire connections (either an output bit or a bit of a
    // multi-bit output wire).
    assert(output.is_wire() || output.as_chunk().is_wire() ||
           output.as_bit().is_wire());
    if ((output.is_chunk() && !output.is_wire()) ||
        ((conn.second.is_chunk() && !conn.second.is_wire()) ||
         conn.second.chunks().size() > 1)) {
      // If one of the RHS or LHS is a chunk of a wire (and not a whole wire) OR
      // contains multiple chunks, then iterate bit by bit to assign the result
      // bits.
      for (auto i = 0; i < output.size(); i++) {
        Value connValue = getBit(conn.second.bits().at(i), b, retBitValues);
        addResultBit(output.bits().at(i), connValue, retBitValues);
      }
    } else {
      // This may be a single bit, a chunk of a wire, or a whole wire.
      Value connValue = getBit(conn.second, b, retBitValues);
      addResultBit(output, connValue, retBitValues);
    }
  }

  // Concatenate result bits if needed, and return result.
  SmallVector<Value, 4> returnValues;
  for (const auto &[resultWire, retBits] : retBitValues) {
    // If we are returning a whole wire as is (e.g. the input wire) or a single
    // bit, we do not need to concat any return bits.
    if (wireNameToValue.contains(resultWire->name.str())) {
      returnValues.push_back(getWireValue(resultWire));
    } else {
      // We are in a multi-bit scenario.
      assert(retBits.size() > 1);
      auto concatOp = b.create<tensor::FromElementsOp>(retBits);
      returnValues.push_back(concatOp.getResult());
    }
  }
  b.create<func::ReturnOp>(returnValues);

  return function;
}

} // namespace mlir
