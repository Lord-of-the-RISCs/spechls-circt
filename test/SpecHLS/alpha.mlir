// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%array: !spechls.array<i32, 4>, %index: i32, %value: i32, %enable: i1) -> !spechls.array<i32, 4> {
  // CHECK: spechls.alpha %arg0[%arg1: i32], %arg2 if %arg3 : !spechls.array<i32, 4>
  %result = spechls.alpha %array[%index: i32], %value if %enable : !spechls.array<i32, 4>
  spechls.commit %result : !spechls.array<i32, 4>
}
