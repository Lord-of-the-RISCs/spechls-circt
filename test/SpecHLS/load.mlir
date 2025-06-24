// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel(%array: !spechls.array<i32, 4>, %index: i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.load %arg0[%arg1 : i32] : <i32, 4>
  %result = spechls.load %array[%index: i32] : !spechls.array<i32, 4>
  spechls.exit if %true with %result : i32
}
