// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : i32) -> !spechls.struct<"out" { "val0" : i32, "val1" : i32 }> {
  %true = hw.constant 1 : i1
  // CHECK: spechls.call @f(%arg0, %arg0) : (i32, i32) -> !spechls.struct<"out" { "val0" : i32, "val1" : i32 }>
  %0 = spechls.call @f(%in1, %in1) : (i32, i32) -> !spechls.struct<"out" { "val0" : i32, "val1" : i32 }>
  spechls.commit %true, %0 : !spechls.struct<"out" { "val0" : i32, "val1" : i32 }>
}
