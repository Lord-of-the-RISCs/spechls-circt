
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU pragma: keep

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <functional>
#include <string>

using namespace mlir;
using namespace circt;

namespace spechls {

#define GEN_PASS_DEF_ARRAYPARTITIONING
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

namespace spechls {

//===----------------------------------------------------------------------===//
// Forward-use traversal
//===----------------------------------------------------------------------===//

/// Traverse the forward use-def chain starting from `start` (included).
/// This is a *graph traversal* following result users. `visit(op)` returns:
///   - true  => continue traversal
///   - false => stop early (and typically signalPassFailure from caller).
static void traverseForwardUses(Operation *start, const std::function<bool(Operation *)> &visit) {
  SmallVector<Operation *, 16> worklist;
  llvm::SmallPtrSet<Operation *, 32> seen;

  worklist.push_back(start);
  seen.insert(start);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visit(op))
      return;

    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (seen.insert(user).second)
          worklist.push_back(user);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Partition bookkeeping
//===----------------------------------------------------------------------===//

/// We map: original array SSA value -> vector of per-partition array SSA values.
/// This is the key fix compared to storing ops: partitioning is about SSA values.
using PartitionVec = SmallVector<Value, 4>;
using PartitionMap = llvm::DenseMap<Value, PartitionVec>;

static FailureOr<const PartitionVec *> lookupPartitions(PartitionMap &pm, Value arrayVal) {

  llvm::outs() << "searching key: " << arrayVal << "::" << arrayVal.getAsOpaquePointer() << "\n";
  // iterate over pm and print keys
  // for (const auto &pair : pm) {
  //   llvm::outs() << "\t- found key: " <<  pair.first << "::"<< pair.first.getAsOpaquePointer() << "\n";
  // }
  auto val = pm[arrayVal];
  auto it = pm.find(arrayVal);
  if (it == pm.end()) {

    llvm::outs() << "key not found " << arrayVal << "::" << arrayVal.getAsOpaquePointer() << "\n";
    return failure();
  } else {
    llvm::outs() << "key found " << arrayVal << "::" << arrayVal.getAsOpaquePointer() << "\n";
    return &it->second;
  }
}

//===----------------------------------------------------------------------===//
// Robust IR building helpers (types + arithmetic)
//===----------------------------------------------------------------------===//

static IntegerType i32Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 32); }

/// Cast `v` to i32 if needed (common denominator for comb ops here).
/// - index -> arith.index_cast
/// - i<N>  -> extui/trunci to i32
static Value castToI32(IRRewriter &rewriter, Location loc, Value v) {
  MLIRContext *ctx = rewriter.getContext();
  Type t = v.getType();

  if (t == i32Ty(ctx))
    return v;

  if (t.isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, i32Ty(ctx), v);

  if (auto intTy = dyn_cast<IntegerType>(t)) {
    unsigned w = intTy.getWidth();
    if (w < 32)
      return rewriter.create<arith::ExtUIOp>(loc, i32Ty(ctx), v);
    if (w > 32)
      return rewriter.create<arith::TruncIOp>(loc, i32Ty(ctx), v);
  }
  // Fallback: keep as-is (verifier will point to the first bad op).
  return v;
}

static Value cstI32(IRRewriter &rewriter, Location loc, int64_t v) {
  return rewriter.create<hw::ConstantOp>(loc, rewriter.getI32IntegerAttr(v)).getResult();
}

/// Block partitioning math:
///   partId = index / block_size   in [0..nb_blocks-1]
///   offset = index % block_size   in [0..block_size-1]
static Value computeBlockPartitionId(IRRewriter &rewriter, Location loc, Value index, int64_t block_size) {
  Value idx32 = castToI32(rewriter, loc, index);
  Value bs32 = cstI32(rewriter, loc, block_size);
  return rewriter.create<comb::DivUOp>(loc, idx32, bs32).getResult();
}

static Value computeBlockOffset(IRRewriter &rewriter, Location loc, Value index, int64_t block_size) {
  Value idx32 = castToI32(rewriter, loc, index);
  Value bs32 = cstI32(rewriter, loc, block_size);
  return rewriter.create<comb::ModUOp>(loc, idx32, bs32).getResult();
}

static Value isPartition(IRRewriter &rewriter, Location loc, Value partIdI32, int64_t idConst) {
  Value idV = cstI32(rewriter, loc, idConst);
  return rewriter.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, partIdI32, idV).getResult();
}

static Value andI1(IRRewriter &rewriter, Location loc, Value a, Value b) {
  return rewriter.create<comb::AndOp>(loc, a, b).getResult();
}

//===----------------------------------------------------------------------===//
// Type building for partitioned arrays
//===----------------------------------------------------------------------===//

static spechls::ArrayType buildPartitionArrayType(spechls::ArrayType arrayTy, int64_t nbBlocks, int64_t blockSize,
                                                  MLIRContext *ctx) {
  assert(nbBlocks > 0 && "nbBlocks must be > 0");
  assert(blockSize > 0 && "blockSize must be > 0");
  // This pass implements equal-size block partitioning; enforce consistency.
  if (arrayTy.getSize() != nbBlocks * blockSize) {
    // Caller will turn this into a proper emitError/signalPassFailure.
    return {};
  }
  return spechls::ArrayType::get(ctx, blockSize, arrayTy.getElementType());
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ArrayPartitioning : public spechls::impl::ArrayPartitioningBase<ArrayPartitioning> {

  ArrayPartitioning() = default;

  // Forward option-based ctor to the generated base if the registration/emit
  // code constructs the pass with options.
  ArrayPartitioning(const spechls::ArrayPartitioningOptions &opts)
      : spechls::impl::ArrayPartitioningBase<ArrayPartitioning>(opts) {}
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *top = getOperation();

    // These are generated by Passes.td options (see updated td snippet below).
    const std::string &targetName = this->arrayName;
    int64_t nbBlocks = this->nbBlocks;
    int64_t blockSizeOpt = this->blockSize;

    if (targetName.empty()) {
      top->emitError("partition-array: option 'array-name' must be provided");
      signalPassFailure();
      return;
    }
    if (nbBlocks <= 0) {
      top->emitError("partition-array: option 'nb-blocks' must be > 0");
      signalPassFailure();
      return;
    }
    if (blockSizeOpt < 0) {
      top->emitError("partition-array: option 'block-size' must be >= 0");
      signalPassFailure();
      return;
    }

    // 1) Find target MuOp by symbol name.
    spechls::MuOp targetMu;
    top->walk([&](spechls::MuOp mu) {
      if (mu.getSymName() == targetName)
        targetMu = mu;
    });

    if (!targetMu) {
      top->emitError(Twine("partition-array: no MuOp found with name '") + targetName + "'");
      signalPassFailure();
      return;
    }

    // 2) Validate types + derive effective blockSize.
    auto arrayTy = dyn_cast<spechls::ArrayType>(targetMu.getResult().getType());
    if (!arrayTy) {
      targetMu.emitError("partition-array: target MuOp result is not ArrayType");
      signalPassFailure();
      return;
    }

    int64_t fullSize = arrayTy.getSize();

    int64_t blockSize = 0;
    if (blockSizeOpt == 0) {
      // Auto mode: require exact divisibility.
      if (fullSize % nbBlocks != 0) {
        targetMu.emitError() << "partition-array: array size (" << fullSize << ") is not divisible by nb-blocks ("
                             << nbBlocks << "); either change nb-blocks or set --block-size explicitly";
        signalPassFailure();
        return;
      }
      blockSize = fullSize / nbBlocks;
    } else {
      blockSize = blockSizeOpt;
      // Enforce exact match: equal-size block partitioning.
      if (blockSize * nbBlocks != fullSize) {
        targetMu.emitError() << "partition-array: invalid (block-size * nb-blocks) = (" << blockSize << " * "
                             << nbBlocks << ") != array size (" << fullSize << ")";
        signalPassFailure();
        return;
      }
    }

    if (blockSize <= 0) {
      targetMu.emitError("partition-array: computed block-size <= 0");
      signalPassFailure();
      return;
    }

    spechls::ArrayType partArrayTy = buildPartitionArrayType(arrayTy, nbBlocks, blockSize, ctx);
    if (!partArrayTy) {
      targetMu.emitError("partition-array: failed to build partition ArrayType");
      signalPassFailure();
      return;
    }

    // 3) Create partition MuOps and seed partitions map.
    IRRewriter rewriter(getOperation());
    rewriter.setInsertionPointAfter(targetMu);

    PartitionMap partitions;

    PartitionVec muParts;
    PartitionVec initParts;
    muParts.reserve(nbBlocks);
    initParts.reserve(nbBlocks);
   auto kernelOp = dyn_cast<KernelOp>(targetMu->getParentOp());

    auto init = targetMu.getInitValue();
    // if init is blokc operand, we partition it too
    auto arg = dyn_cast<BlockArgument>(init);
    if (arg != nullptr) {
      // add argument to block
      for (int64_t i = 0; i < nbBlocks; ++i) {
        auto newarg = init.getParentBlock()->addArgument(partArrayTy, init.getLoc());

        initParts.push_back(newarg);
      }

      // 1. Update function type

      if (kernelOp == nullptr) {
        targetMu.emitError("MuOp has no parent function");
        signalPassFailure();
        return;
      }
      FunctionType oldType = kernelOp.getFunctionType();
      if (oldType == nullptr) {
        targetMu.emitError("parent op has not a Function type");
        signalPassFailure();
        return;
      }
      SmallVector<Type> newInputs(oldType.getInputs().begin(), oldType.getInputs().end());
      for (int64_t i = 0; i < nbBlocks; ++i) {
        newInputs.push_back(partArrayTy);
      }
      auto newType = FunctionType::get(kernelOp->getContext(), newInputs, oldType.getResults());

      kernelOp.setFunctionType(newType);

    } else {
      targetMu.emitError("invalid pattern for partitioning");
      signalPassFailure();
    }

    for (int64_t i = 0; i < nbBlocks; ++i) {
      std::string newSym = (targetMu.getSymName().str() + "_part_" + std::to_string(i));

      // NOTE: This assumes MuOp builder is (loc, resultTy, symName, init, loop).
      // If your MuOp signature differs, adjust accordingly.
      auto newMu =
          rewriter.create<spechls::MuOp>(targetMu.getLoc(), partArrayTy, newSym, initParts[i], targetMu.getLoopValue());

      // newMu.verify();
      muParts.push_back(newMu.getResult());
      llvm::outs() << "Created partitioned MuOp: " << newMu << "\n";
    }
    partitions[targetMu.getResult()] = muParts;

    // 4) collect forward users.
    llvm::SmallVector<Operation *, 32> users;
    llvm::SmallVector<Operation *, 32> deadOps;

    deadOps.push_back(targetMu);

    traverseForwardUses(targetMu.getOperation(), [&](Operation *o) -> bool {
      users.push_back(o);
      llvm::outs() << "Found user op: " << *o << "\n";
      return true;
    }); // 4) Rewrite forward users.

    for (auto o : users) {
      llvm::outs() << "dispatch on: " << *o << "\n";
      llvm::TypeSwitch<Operation *>(o)
          .Case<spechls::AlphaOp>([&](spechls::AlphaOp alpha) {
            auto partsOrFail = lookupPartitions(partitions, alpha.getArray());
            if (failed(partsOrFail)) {
              alpha.emitError("partition-array: no mapping for alpha.getArray()");
              signalPassFailure();
            }
            const PartitionVec &inParts = **partsOrFail;

            PartitionVec outParts;
            outParts.reserve(nbBlocks);

            Location loc = alpha.getLoc();
            Value partId = computeBlockPartitionId(rewriter, loc, alpha.getIndex(), blockSize);
            Value offset = computeBlockOffset(rewriter, loc, alpha.getIndex(), blockSize);

            for (int64_t i = 0; i < nbBlocks; ++i) {
              Value isThis = isPartition(rewriter, loc, partId, i);
              Value we_i = andI1(rewriter, loc, alpha.getWe(), isThis);

              auto newAlpha =
                  rewriter.create<spechls::AlphaOp>(loc, partArrayTy, inParts[i], alpha.getValue(), offset, we_i);
              llvm::outs() << "Created partitioned AlphaOp: " << newAlpha << "\n";
              outParts.push_back(newAlpha.getResult());
            }

            partitions[alpha.getResult()] = outParts;
            deadOps.push_back(alpha);
          })
          .Case<spechls::LoadOp>([&](spechls::LoadOp load) {
            // Load: read from array. Load each partition at offset, then Gamma select.
            llvm::outs() << "Search partition for LoadOp: " << load.getArray() << "\n";
            auto partsOrFail = lookupPartitions(partitions, load.getArray());
            if (failed(partsOrFail)) {
              load.emitError("partition-array: no mapping for load.getArray()");
              signalPassFailure();
            }
            const PartitionVec &inParts = **partsOrFail;

            Location loc = load.getLoc();

            Value partId = computeBlockPartitionId(rewriter, loc, load.getIndex(), blockSize);
            Value offset = computeBlockOffset(rewriter, loc, load.getIndex(), blockSize);

            SmallVector<Value, 8> perPartLoads;
            perPartLoads.reserve(nbBlocks);

            for (int64_t i = 0; i < nbBlocks; ++i) {
              auto newLoad = rewriter.create<spechls::LoadOp>(loc, load.getResult().getType(), inParts[i], offset);
              perPartLoads.push_back(newLoad.getResult());
              llvm::outs() << "Created partitioned LoadOp: " << newLoad << "\n";
            }

            auto gamma = rewriter.create<spechls::GammaOp>(loc, load.getResult().getType(),
                                                           rewriter.getStringAttr("load_part_select"), partId,
                                                           ValueRange(perPartLoads));
            llvm::outs() << "Created merging GammaOp: " << gamma << "\n";
            load.replaceAllUsesWith(gamma.getResult());
            deadOps.push_back(load);
          })
          .Case<spechls::SyncOp>([&](spechls::SyncOp sync) {
            // Sync: expand operands that are partitioned into all partitions.
            Location loc = sync.getLoc();
            SmallVector<Value, 16> newOperands;
            llvm::SmallPtrSet<Value, 16> seen;

            for (Value opnd : sync.getOperands()) {
              auto it = partitions.find(opnd);
              if (it == partitions.end()) {
                if (seen.insert(opnd).second)
                  newOperands.push_back(opnd);
                continue;
              }
              for (Value pv : it->second) {
                if (seen.insert(pv).second)
                  newOperands.push_back(pv);
              }
            }

            auto newSync = rewriter.create<spechls::SyncOp>(loc, newOperands);
            sync.replaceAllUsesWith(newSync.getResult());
            deadOps.push_back(sync);
          })
          .Default([&](Operation *other) { llvm::outs() << "unexpected op " << *other; });
   }

   for (int i=0;i<nbBlocks;i++) {
      Value mu = muParts[i];

        llvm::outs() << "\n\nPartitioned MuOp value: " << mu << "\n";
      auto muOp = dyn_cast<MuOp>(mu.getDefiningOp());
      if (muOp!=nullptr) {
        llvm::outs() << "MuOp details: " << muOp << "\n";
        auto partsOrFail = lookupPartitions(partitions, muOp.getLoopValue());
        if (failed(partsOrFail)) {
          muOp.emitError("partition-array: no mapping for load.getArray()");
          signalPassFailure();
        }
        const PartitionVec &inParts = **partsOrFail;
        for (size_t j=0;j<inParts.size();j++)
          llvm::outs() << "\t- init part " << j << ": " << inParts[j] << "\n";
        muOp.setOperand(1, inParts[i]);
      }
   }

    targetMu.setOperand(1, init); // restore original init to target MuOp

    bool changed = true;


    // remove SCC dead ops
    while (changed) {
      changed = false;

      // Index loop so we can remove elements safely.
      for (size_t i = 0; i < deadOps.size(); ) {
        mlir::Operation *op = deadOps[i];

        if (op && op->use_empty()) {
          rewriter.eraseOp(op);

          // Remove op from deadOps in O(1) by swapping with the last element.
          deadOps[i] = deadOps.back();
          deadOps.pop_back();

          changed = true;
          // don't ++i because we need to process the element we swapped in
        } else {
          ++i;
        }
      }
    }
    llvm::outs() << kernelOp << "\n";

     llvm::outs() << "End of pass\n";
  }
};

} // namespace spechls
