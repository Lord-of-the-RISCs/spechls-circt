// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%array: !spechls.array<i32, 4>, %index: i32) -> i32 {
  // CHECK: spechls.alpha %arg0[%arg1: i32], %arg2 if %arg3 : !spechls.array<i32, 4>
  %result = spechls.load %array[%index: i32] : !spechls.array<i32, 4>
  spechls.commit %result : i32
}
