// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel(%array: !spechls.array<i32, 4>, %index: i32, %value: i32, %enable: i1) -> !spechls.array<i32, 4> {
  %true = hw.constant 1 : i1
  // CHECK: spechls.alpha %arg0[%arg1: i32], %arg2 if %arg3 : !spechls.array<i32, 4>
  %result = spechls.alpha %array[%index: i32], %value if %enable : !spechls.array<i32, 4>
  spechls.exit if %true with %result : !spechls.array<i32, 4>
}
