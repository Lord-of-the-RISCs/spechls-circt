// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : i32) -> !spechls.struct<i32, i32> {
  // CHECK: spechls.call @f(%arg0, %arg0) : (i32, i32) -> !spechls.struct<i32, i32>
  %0 = spechls.call @f(%in1, %in1) : (i32, i32) -> !spechls.struct<i32, i32>
  spechls.commit %0 : !spechls.struct<i32, i32>
}
