// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.task @task(%in1 : !spechls.struct<"in" { "val0" : i1, "val1" : !spechls.struct<"out" { "val0" : i1, "val1" : i32 }> }>) -> !spechls.struct<"out" { "val0" : i1, "val1" : i32 }> {
  %true = hw.constant 1 : i1
  // CHECK: spechls.fifo<4> %arg0 : (!spechls.struct<"in" { "val0" : i1, "val1" : !spechls.struct<"out" { "val0" : i1, "val1" : i32 }> }>) -> !spechls.struct<"out" { "val0" : i1, "val1" : i32 }>
  %0 = spechls.fifo<4> %in1 : (!spechls.struct<"in" { "val0" : i1, "val1" : !spechls.struct<"out" { "val0" : i1, "val1" : i32 }> }>) -> !spechls.struct<"out" { "val0" : i1, "val1" : i32 }>
  spechls.commit %true, %0 : !spechls.struct<"out" { "val0" : i1, "val1" : i32 }>
}
