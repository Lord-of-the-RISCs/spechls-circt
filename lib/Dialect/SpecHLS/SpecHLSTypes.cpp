//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSTypes.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "Dialect/SpecHLS/SpecHLSOpsTypes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "Dialect/SpecHLS/SpecHLSOpsTypes.cpp.inc"

namespace SpecHLS {

mlir::Type SpecArrayType::parse(mlir::AsmParser &parser) {
  Type baseType;
  int64_t nbDiscardedWrites;
  int64_t maxPendingAddresses;
  int64_t maxPendingWrites;

  SmallVector<int64_t, 16> nbpendingWrites;
  ParseResult nok = parser.parseLess();
  nok = parser.parseLess();

  nok = parser.parseDimensionList(nbpendingWrites);

  nok = parser.parseType(baseType);

  nok = parser.parseGreater();

  nok = parser.parseLBrace();
  nok = parser.parseInteger(nbDiscardedWrites);
  nok = parser.parseComma();

  nok = parser.parseLBrace();
  do {
    int64_t tmp;
    nok = parser.parseInteger(tmp);
    nbpendingWrites.push_back(tmp);
    nok = parser.parseOptionalComma();
  } while (!nok);
  nok = parser.parseRBrace();

  nok = parser.parseComma();
  nok = parser.parseInteger(maxPendingWrites);

  nok = parser.parseComma();
  nok = parser.parseInteger(maxPendingAddresses);

  nok = parser.parseRBrace();

  // parser.getBuilder()
}

/*
 "int64_t":$size,
"Type":$elementType,
"int64_t":$nbDiscardedWrites,
ArrayRefParameter<"int64_t">:$nbPendingWrites,
"int64_t":$maxPendingWrites,
"int64_t":$maxPendingAddresses
 */

// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void SpecArrayType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getSize() << "x" << getElementType() << ">";

  printer << "{" << (getNbDiscardedWrites());

  printer << "{" << (getNbDiscardedWrites());
  for (size_t k = 0; k < getNbPendingWrites().size(); k++) {
    if (k > 0)
      printer << ", ";
    printer << getNbPendingWrites()[k];
  }
  printer << "},";

  printer << (getMaxPendingWrites()) << ",";
  printer << (getMaxPendingAddresses()) << "}";
}

} // namespace SpecHLS
