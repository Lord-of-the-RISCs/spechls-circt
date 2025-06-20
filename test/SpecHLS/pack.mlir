// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.task @task(%in1 : !spechls.struct<"in" { "val0" : i32, "val1" : i16, "val2" : i1 }>) -> !spechls.struct<"out" { "val0" : i1, "val1" : i16, "val2" : i32 }> {
  %true = hw.constant 1 : i1
  // CHECK: spechls.unpack %arg0 : <"in" { "val0" : i32, "val1" : i16, "val2" : i1 }>
  // CHECK: spechls.pack %0#2, %0#1, %0#0 : (i1, i16, i32) -> !spechls.struct<"out" { "val0" : i1, "val1" : i16, "val2" : i32 }>
  %0:3 = spechls.unpack %in1 : !spechls.struct<"in" { "val0" : i32, "val1" : i16, "val2" : i1 }>
  %1 = spechls.pack %0#2, %0#1, %0#0 : (i1, i16, i32) -> !spechls.struct<"out" { "val0" : i1, "val1" : i16, "val2" : i32 }>
  spechls.commit %true, %1 : !spechls.struct<"out" { "val0" : i1, "val1" : i16, "val2" : i32 }>
}
