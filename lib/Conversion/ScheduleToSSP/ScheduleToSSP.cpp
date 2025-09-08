//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/Passes.h" // IWYU pragma: keep
#include "Dialect/Schedule/IR/ScheduleOps.h"

#include <circt/Dialect/SSP/SSPAttributes.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <circt/Dialect/SSP/SSPOps.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace schedule {
#define GEN_PASS_DEF_SCHEDULETOSSPPASS
#include "Conversion/Passes.h.inc"
}; // namespace schedule

using namespace circt;
using namespace mlir;

namespace {

struct ScheduleToSSPOpConversion : OpConversionPattern<schedule::CircuitOp> {
  using OpConversionPattern<schedule::CircuitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(schedule::CircuitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    rewriter.setInsertionPointAfter(op);
    auto instanceOp = rewriter.create<ssp::InstanceOp>(op->getLoc(), rewriter.getStringAttr("ChainingCyclicProblem"));

    rewriter.setInsertionPointToStart(instanceOp.getBodyBlock());
    auto library = rewriter.create<ssp::OperatorLibraryOp>(op.getLoc());
    auto graph = rewriter.create<ssp::DependenceGraphOp>(op.getLoc());

    instanceOp->setAttr("spechls.period", op.getTargetClockAttr());

    llvm::DenseMap<schedule::OperationOp, ssp::OperationOp> operationMap;
    int idx = 0;

    for (auto &&operation : op.getBody().front()) {
      if (auto scheduleOp = dyn_cast<schedule::OperationOp>(operation)) {
        std::string operatorName = "operator_" + std::to_string(idx++);
        ;
        llvm::SmallVector<mlir::Attribute> operatorPropertyValues;

        operatorPropertyValues.push_back(
            ssp::LatencyAttr::get(rewriter.getContext(), scheduleOp.getLatencyAttr().getInt()));
        operatorPropertyValues.push_back(
            ssp::IncomingDelayAttr::get(rewriter.getContext(), scheduleOp.getInDelayAttr()));
        operatorPropertyValues.push_back(
            ssp::OutgoingDelayAttr::get(rewriter.getContext(), scheduleOp.getOutDelayAttr()));

        rewriter.setInsertionPointToEnd(library.getBodyBlock());
        rewriter.create<ssp::OperatorTypeOp>(op.getLoc(), operatorName, rewriter.getArrayAttr(operatorPropertyValues));

        ::llvm::SmallVector<mlir::Attribute> deps;
        ::llvm::SmallVector<mlir::Value> operands;

        unsigned int opId = 0;

        for (size_t i = 0; i < scheduleOp.getDependences().size(); ++i) {
          unsigned distance = cast<IntegerAttr>(scheduleOp.getDistances()[i]).getInt();
          if (distance == 0) {
            operands.push_back(
                operationMap[cast<schedule::OperationOp>(*scheduleOp.getOperand(i).getDefiningOp())]->getResult(0));
            ++opId;
          }
        }

        for (size_t i = 0; i < scheduleOp.getDependences().size(); ++i) {
          unsigned int distance = cast<IntegerAttr>(scheduleOp.getDistances()[i]).getInt();
          ::llvm::SmallVector<::mlir::Attribute> distanceAttrArray;
          distanceAttrArray.push_back(ssp::DistanceAttr::get(rewriter.getContext(), distance));
          if (distance != 0) {
            deps.push_back(ssp::DependenceAttr::get(
                rewriter.getContext(), opId++,
                ::mlir::FlatSymbolRefAttr::get(rewriter.getStringAttr(
                    std::to_string(reinterpret_cast<uintptr_t>(scheduleOp.getOperand(i).getDefiningOp())))),
                // cast<schedule::OperationOp>(scheduleOp.getOperand(i).getDefiningOp()).getSymName())),
                rewriter.getArrayAttr(distanceAttrArray)));
          }
        }
        rewriter.setInsertionPointToEnd(graph.getBodyBlock());
        auto newOperation = rewriter.create<ssp::OperationOp>(
            scheduleOp->getLoc(), 1, operands,
            rewriter.getStringAttr(std::to_string(reinterpret_cast<uintptr_t>(scheduleOp.getOperation()))),
            rewriter.getArrayAttr(deps));

        llvm::SmallVector<mlir::Attribute> properties;
        properties.push_back(ssp::LinkedOperatorTypeAttr::get(rewriter.getContext(),
                                                              rewriter.getAttr<::mlir::SymbolRefAttr>(operatorName)));

        newOperation.setSspPropertiesAttr(rewriter.getArrayAttr(properties));

        operationMap.try_emplace(scheduleOp, newOperation);

      } else {
        llvm::errs() << "schedule::circuit body contains illegal operation :\n";
        operation.dump();
        return failure();
      }
    }
    return success();
  }
};

}; // namespace

namespace {

struct ConvertScheduleToSSPPass : public schedule::impl::ScheduleToSSPPassBase<ConvertScheduleToSSPPass> {
  void runOnOperation() override;
  using ScheduleToSSPPassBase<ConvertScheduleToSSPPass>::ScheduleToSSPPassBase;
};

void populateScheduleToSSPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ScheduleToSSPOpConversion>(patterns.getContext());
}

}; // namespace

void ConvertScheduleToSSPPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<ssp::SSPDialect>();
  //  target.addIllegalDialect<schedule::ScheduleDialect>();

  RewritePatternSet patterns(&getContext());
  populateScheduleToSSPConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
