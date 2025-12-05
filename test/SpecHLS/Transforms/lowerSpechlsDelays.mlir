// RUN: spechls-opt -split-input-file --lower-spechls-delays %s | spechls-opt | FileCheck %s

spechls.kernel @kernelRb1(%in1 : i1, %rollback : i32, %rbwe :i1) -> i1 {
  %true = hw.constant true
  // CHECK: %0 = spechls.delay %arg0 by 1 {} : i1
  // CHECK: %1 = spechls.rollback<[1, 2], 2> %0, %arg1, %arg2 : i1, i32
  %0 = spechls.rollbackableDelay %in1 by 1 rollback %rollback : i32 at 2 %rbwe [1, 2]: i1
  spechls.exit if %true with %0 : i1
}

spechls.kernel @kernelRb2(%in1 : i1, %rbwe :i1) -> i1 {
  %true = hw.constant true
  %false = hw.constant false
  // CHECK: %0 = spechls.delay %arg0 by 2 {} : i1
  %0 = spechls.rollbackableDelay %in1 by 2 rollback %false : i1 at 2 %rbwe [3,5,7]: i1
  spechls.exit if %true with %0 : i1
}

spechls.kernel @kernelCl1(%in1 : i1, %cancel : i32, %clwe :i1) -> i1 {
  %true = hw.constant true
  // CHECK: %0 = spechls.delay %arg0 by 1 {} : i1
  // CHECK: %1 = spechls.cancel<2> %0, %arg1, %arg2 : i32
  %0 = spechls.cancellableDelay %in1 by 1 cancel %cancel : i32 at 2 %clwe : i1
  spechls.exit if %true with %0 : i1
}

spechls.kernel @kernelCl2(%in1 : i1, %clwe :i1) -> i1 {
  %true = hw.constant true
  %false = hw.constant false
  // CHECK: %0 = spechls.delay %arg0 by 7 {} : i1
  %0 = spechls.cancellableDelay %in1 by 7 cancel %false : i1 at 2 %clwe : i1
  spechls.exit if %true with %0 : i1
}