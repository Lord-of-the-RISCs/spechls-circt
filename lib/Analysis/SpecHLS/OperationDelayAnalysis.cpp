//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/OperationDelayAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <fstream>
#include <sstream>
#include <string>

namespace spechls {

NotOperatorTiming::NotOperatorTiming() : cstValue(0.0) {}
NotOperatorTiming::NotOperatorTiming(llvm::json::Value *jsonValue) {
  if (jsonValue->kind() != llvm::json::Value::Number) {
    llvm::errs() << "Wrong value type for operator not.\n";
    exit(1);
  }
  cstValue = jsonValue->getAsNumber().value();
}

AndOperatorTiming::AndOperatorTiming() : linearParameter(0.0), cstParameter(0.0) {}

AndOperatorTiming::AndOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  linearParameter = array->data()[0].getAsNumber().value();
  cstParameter = array->data()[1].getAsNumber().value();
}

OrOperatorTiming::OrOperatorTiming() : linearParameter(0.0), cstParameter(0.0) {}

OrOperatorTiming::OrOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  linearParameter = array->data()[0].getAsNumber().value();
  cstParameter = array->data()[1].getAsNumber().value();
}

XorOperatorTiming::XorOperatorTiming() : linearParameter(0.0), cstParameter(0.0) {}

XorOperatorTiming::XorOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  linearParameter = array->data()[0].getAsNumber().value();
  cstParameter = array->data()[1].getAsNumber().value();
}

MuxOperatorTiming::MuxOperatorTiming() : cstParameter(0.0), bwParameter(0.0), ctrlBwParameter(0.0) {}

MuxOperatorTiming::MuxOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  cstParameter = array->data()[0].getAsNumber().value();
  bwParameter = array->data()[1].getAsNumber().value();
  ctrlBwParameter = array->data()[1].getAsNumber().value();
}

EqOperatorTiming::EqOperatorTiming() : cstParameter(0.0), bwParameter(0.0) {}

EqOperatorTiming::EqOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  cstParameter = array->data()[0].getAsNumber().value();
  bwParameter = array->data()[1].getAsNumber().value();
}

AddOperatorTiming::AddOperatorTiming() : cstParameter(0.0), bwParameter(0.0) {}

AddOperatorTiming::AddOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  cstParameter = array->data()[0].getAsNumber().value();
  bwParameter = array->data()[1].getAsNumber().value();
}

ShiftOperatorTiming::ShiftOperatorTiming() : cstParameter(0.0), bwParameter(0.0) {}

ShiftOperatorTiming::ShiftOperatorTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  cstParameter = array->data()[0].getAsNumber().value();
  bwParameter = array->data()[1].getAsNumber().value();
}

LocalMemoryTiming::LocalMemoryTiming() : inDelay(0.0), outDelay(0.0), latency(0) {}

LocalMemoryTiming::LocalMemoryTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  inDelay = array->data()[0].getAsNumber().value();
  latency = array->data()[1].getAsInteger().value();
  outDelay = array->data()[2].getAsNumber().value();
}

ExternalMemoryTiming::ExternalMemoryTiming() : inDelay(0.0), outDelay(0.0), latency(0) {}

ExternalMemoryTiming::ExternalMemoryTiming(llvm::json::Value *jsonValue) {
  auto *array = jsonValue->getAsArray();
  inDelay = array->data()[0].getAsNumber().value();
  latency = array->data()[1].getAsInteger().value();
  outDelay = array->data()[2].getAsNumber().value();
}

TimingAnalyser::TimingAnalyser(llvm::json::Object *targetJson) {
  if (auto *notOp = targetJson->get("not")) {
    notOpTiming = NotOperatorTiming(notOp);
  } else {
    notOpTiming = NotOperatorTiming{};
    llvm::errs() << "No timing value for not operator.\n";
  }
  if (auto *andOp = targetJson->get("and")) {
    andOpTiming = AndOperatorTiming(andOp);
  } else {
    andOpTiming = AndOperatorTiming{};
    llvm::errs() << "No timing value for and operator.\n";
  }
  if (auto *orOp = targetJson->get("or")) {
    orOpTiming = OrOperatorTiming(orOp);
  } else {
    orOpTiming = OrOperatorTiming{};
    llvm::errs() << "No timing value for or operator.\n";
  }
  if (auto *xorOp = targetJson->get("xor")) {
    xorOpTiming = XorOperatorTiming(xorOp);
  } else {
    xorOpTiming = XorOperatorTiming{};
    llvm::errs() << "No timing value for xor operator.\n";
  }
  if (auto *muxOp = targetJson->get("mux")) {
    muxOpTiming = MuxOperatorTiming(muxOp);
  } else {
    muxOpTiming = MuxOperatorTiming{};
    llvm::errs() << "No timing value for mux operator.\n";
  }
  if (auto *eqOp = targetJson->get("eq")) {
    eqOpTiming = EqOperatorTiming(eqOp);
  } else {
    eqOpTiming = EqOperatorTiming{};
    llvm::errs() << "No timing value for eq operator.\n";
  }
  if (auto *addOp = targetJson->get("add")) {
    addOpTiming = AddOperatorTiming(addOp);
  } else {
    addOpTiming = AddOperatorTiming{};
    llvm::errs() << "No timing value for add operator.\n";
  }
  if (auto *shiftOp = targetJson->get("shift")) {
    shiftOpTiming = ShiftOperatorTiming(shiftOp);
  } else {
    shiftOpTiming = ShiftOperatorTiming{};
    llvm::errs() << "No timing value for shift operator.\n";
  }
  if (auto *localMemory = targetJson->get("localMemory")) {
    localMemTiming = LocalMemoryTiming(localMemory);
  } else {
    localMemTiming = LocalMemoryTiming{};
    llvm::errs() << "No timing value for local memory.\n";
  }
  if (auto *externalMemory = targetJson->get("externalMemory")) {
    externalMemTiming = ExternalMemoryTiming(externalMemory);
  } else {
    externalMemTiming = ExternalMemoryTiming{};
    llvm::errs() << "No timing value for global memory.\n";
  }
}

TimingAnalyser::TimingAnalyser()
    : notOpTiming{}, andOpTiming{}, orOpTiming{}, xorOpTiming{}, muxOpTiming{}, eqOpTiming{}, addOpTiming{},
      shiftOpTiming{}, localMemTiming{}, externalMemTiming{} {}

TimingAnalyserFactory::TimingAnalyserFactory() : defaultAnalyser{} {}

mlir::LogicalResult TimingAnalyserFactory::registerAnalyers(std::string jsonFile) {
  std::ifstream file(jsonFile);
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file " << jsonFile << ".\n";
    return mlir::failure();
  }
  std::stringstream content;
  content << file.rdbuf();
  file.close();
  auto json = llvm::json::parse(content.str());
  if (auto err = json.takeError()) {
    llvm::handleAllErrors(std::move(err), [&](const llvm::json::ParseError &a) {
      llvm::errs() << ": error parsing view JSON: " << a.message();
    });
    return mlir::failure();
  }
  if (json->kind() != llvm::json::Value::Kind::Object) {
    llvm::errs() << "Wrong json object type.\n";
    return mlir::failure();
  }
  auto *jsonObject = json->getAsObject();
  for (auto &[target, value] : *jsonObject) {
    if (analysers.contains(target.str()))
      continue;

    if (value.kind() != llvm::json::Value::Kind::Object) {
      llvm::errs() << "Wrong json object type for target " << target << ".\n";
      return mlir::failure();
    }

    auto *valueObject = value.getAsObject();

    analysers.try_emplace(target, TimingAnalyser(valueObject));
  }
  return mlir::success();
}

TimingAnalyser TimingAnalyserFactory::get(std::string target) {
  if (analysers.contains(target))
    return analysers.lookup(target);
  return defaultAnalyser;
}

TimingAnalyserFactory timingAnalyserFactory;

std::optional<std::tuple<double, int, double>>
TimingAnalyser::computeOperationTiming(mlir::Operation *op, double clockPeriod, mlir::MLIRContext *ctx) {
  if (auto combDelayAttr = op->getAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
    return std::make_tuple(combDelayAttr.getValueAsDouble(), 0, combDelayAttr.getValueAsDouble());
  }
  if (op->hasAttr("spechls.latency")) {
    unsigned latency = op->getAttrOfType<mlir::IntegerAttr>("spechls.lantency").getUInt();
    return std::make_tuple(0.0, latency, 0.0);
  }
  return llvm::TypeSwitch<mlir::Operation *, std::optional<std::tuple<double, int, double>>>(op)
      .Case<circt::comb::AddOp, circt::comb::SubOp>([&](mlir::Operation *op) {
        double bw = static_cast<double>(op->getResultTypes().front().getIntOrFloatBitWidth());
        double combDelay = this->addOpTiming.cstParameter + this->addOpTiming.bwParameter * bw;
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case<spechls::LoadOp, spechls::AlphaOp>([&](mlir::Operation *op) {
        return std::make_tuple(this->localMemTiming.inDelay, this->localMemTiming.latency,
                               this->localMemTiming.outDelay);
      })
      .Case<circt::comb::ICmpOp>([&](mlir::Operation *op) {
        double bw = static_cast<double>(op->getResultTypes().front().getIntOrFloatBitWidth());
        double combDelay = this->eqOpTiming.cstParameter + this->eqOpTiming.bwParameter * std::log2(bw);
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case([&](circt::comb::MuxOp &op) {
        double bw = static_cast<double>(op->getResultTypes().front().getIntOrFloatBitWidth());
        double ctrlBw = static_cast<double>(op.getCond().getType().getIntOrFloatBitWidth());
        double combDelay = this->muxOpTiming.cstParameter + this->muxOpTiming.bwParameter * std::log2(bw) +
                           this->muxOpTiming.ctrlBwParameter * ctrlBw;
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case([&](spechls::GammaOp &op) {
        if (llvm::isa<spechls::ArrayType>(op->getResult(0).getType())) {
          return std::make_tuple(0.0, 0, 0.0);
        }
        double bw = static_cast<double>(op->getResultTypes().front().getIntOrFloatBitWidth());
        double ctrlBw = static_cast<double>(op.getOperand(0).getType().getIntOrFloatBitWidth());
        double combDelay = this->muxOpTiming.cstParameter + this->muxOpTiming.bwParameter * std::log2(bw) +
                           this->muxOpTiming.ctrlBwParameter * ctrlBw;
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case<circt::comb::ShlOp, circt::comb::ShrSOp, circt::comb::ShrUOp>([&](mlir::Operation *op) {
        double combDelay = 0.0;
        if (!llvm::isa<circt::hw::ConstantOp>(op->getOperand(1).getDefiningOp())) {
          double bw = static_cast<double>(op->getResultTypes().front().getIntOrFloatBitWidth());
          combDelay = this->shiftOpTiming.cstParameter + this->shiftOpTiming.bwParameter * bw;
        }
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case<circt::hw::ConstantOp, circt::hw::BitcastOp, spechls::MuOp, spechls::DelayOp, spechls::RollbackableDelayOp,
            spechls::CancellableDelayOp, circt::comb::ExtractOp, circt::comb::ConcatOp, spechls::TaskOp,
            spechls::PrintOp, spechls::ExitOp, spechls::KernelOp, spechls::SyncOp, spechls::PackOp,
            spechls::FSMCommandOp, spechls::FSMOp, circt::comb::ReplicateOp>(
          [&](mlir::Operation *op) { return std::make_tuple(0.0, 0, 0.0); })
      .Case([&](circt::comb::OrOp &op) {
        double combDelay = this->orOpTiming.cstParameter + this->orOpTiming.linearParameter * op.getNumOperands();
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case([&](circt::comb::XorOp &op) {
        double combDelay = this->xorOpTiming.cstParameter + this->xorOpTiming.linearParameter * op.getNumOperands();
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case([&](circt::comb::AndOp &op) {
        double combDelay = this->andOpTiming.cstParameter + this->andOpTiming.linearParameter * op.getNumOperands();
        return std::make_tuple(combDelay, 0, combDelay);
      })
      .Case<spechls::CallOp, spechls::FieldOp, spechls::CommitOp>(
          [&](mlir::Operation *op) { return std::make_tuple(0.0, 0, 0.0); })
      .Default([&](mlir::Operation *op) {
        llvm::errs() << "No timing found for following operation.\n";
        op->dump();
        return std::nullopt;
      });
}

} // namespace spechls