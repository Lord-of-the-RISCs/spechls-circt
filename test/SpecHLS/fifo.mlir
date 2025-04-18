// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : !spechls.struct<"in" { "val0" : i1, "val1" : i32 }>) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.fifo<4> %arg0 : (!spechls.struct<"in" { "val0" : i1, "val1" : i32 }>) -> i32
  %0 = spechls.fifo<4> %in1 : (!spechls.struct<"in" { "val0" : i1, "val1" : i32 }>) -> i32
  spechls.commit %true, %0 : i32
}
