//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_ANALYSIS_OPERATION_DELAY_ANALYSIS__
#define SPECHLS_ANALYSIS_OPERATION_DELAY_ANALYSIS__

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/JSON.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <string>

namespace spechls {

struct NotOperatorTiming {
  double cstValue;
  NotOperatorTiming();
  NotOperatorTiming(llvm::json::Value *jsonValue);
};

struct AndOperatorTiming {
  double linearParameter, cstParameter;
  AndOperatorTiming();
  AndOperatorTiming(llvm::json::Value *jsonValue);
};

struct OrOperatorTiming {
  double linearParameter, cstParameter;
  OrOperatorTiming();
  OrOperatorTiming(llvm::json::Value *jsonValue);
};

struct XorOperatorTiming {
  double linearParameter, cstParameter;
  XorOperatorTiming();
  XorOperatorTiming(llvm::json::Value *jsonValue);
};

struct MuxOperatorTiming {
  double cstParameter, bwParameter, ctrlBwParameter;
  MuxOperatorTiming();
  MuxOperatorTiming(llvm::json::Value *jsonValue);
};

struct EqOperatorTiming {
  double cstParameter, bwParameter;
  EqOperatorTiming();
  EqOperatorTiming(llvm::json::Value *jsonValue);
};

struct AddOperatorTiming {
  double cstParameter, bwParameter;
  AddOperatorTiming();
  AddOperatorTiming(llvm::json::Value *jsonValue);
};

struct ShiftOperatorTiming {
  double cstParameter, bwParameter;
  ShiftOperatorTiming();
  ShiftOperatorTiming(llvm::json::Value *jsonValue);
};

struct LocalMemoryTiming {
  double inDelay, outDelay;
  int latency;
  LocalMemoryTiming();
  LocalMemoryTiming(llvm::json::Value *jsonValue);
};

struct ExternalMemoryTiming {
  double inDelay, outDelay;
  int latency;
  ExternalMemoryTiming();
  ExternalMemoryTiming(llvm::json::Value *jsonValue);
};

struct TimingAnalyser {
  NotOperatorTiming notOpTiming;
  AndOperatorTiming andOpTiming;
  OrOperatorTiming orOpTiming;
  XorOperatorTiming xorOpTiming;
  MuxOperatorTiming muxOpTiming;
  EqOperatorTiming eqOpTiming;
  AddOperatorTiming addOpTiming;
  ShiftOperatorTiming shiftOpTiming;
  LocalMemoryTiming localMemTiming;
  ExternalMemoryTiming externalMemTiming;

  TimingAnalyser();
  TimingAnalyser(llvm::json::Object *targetJson);

  std::optional<std::tuple<double, int, double>> computeOperationTiming(mlir::Operation *op, double clockPeriod,
                                                                        mlir::MLIRContext *ctx);
};

struct TimingAnalyserFactory {
  llvm::StringMap<TimingAnalyser> analysers;
  TimingAnalyser defaultAnalyser;

  TimingAnalyserFactory();
  TimingAnalyser get(std::string target);

  mlir::LogicalResult registerAnalyers(std::string jsonFile);
};

extern TimingAnalyserFactory timingAnalyserFactory;

} // namespace spechls

#endif // SPECHLS_ANALYSIS_OPERATION_DELAY_ANALYSIS__