// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
spechls.kernel @kernel(%in1 : i1, %in2 : i1, %cancel : i1) -> i1 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.cancellableDelay %arg0 by 1 cancel %arg2 at 0 %true {} : i1
  // CHECK: spechls.cancellableDelay %0 by 2 cancel %arg2 at 1 %true if %arg1 {} : i1
  // CHECK: spechls.cancellableDelay %1 by 1 cancel %arg2 at 0 %true init %arg0 {} : i1
  // CHECK: spechls.cancellableDelay %2 by 4 cancel %arg2 at 3 %true if %arg1 init %arg0 {} : i1
  // CHECK: spechls.cancellableDelay %2 by 4 cancel %arg2 at 2 %true if %arg1 init %4 {} : i1
  %d0 = spechls.cancellableDelay %in1 by 1 cancel %cancel at 0 %true : i1
  %d1 = spechls.cancellableDelay %d0 by 2 cancel %cancel at 1 %true if %in2 : i1
  %d2 = spechls.cancellableDelay %d1 by 1 cancel %cancel at 0 %true init %in1 : i1
  %d3 = spechls.cancellableDelay %d2 by 4 cancel %cancel at 3 %true if %in2 init %in1 : i1
  %d4 = spechls.cancellableDelay %d2 by 4 cancel %cancel at 2 %true if %in2 init %d4 : i1
  spechls.exit if %in2 with %d4 : i1
}
