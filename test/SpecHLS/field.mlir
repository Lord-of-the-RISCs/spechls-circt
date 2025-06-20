// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.task @task(%in : !spechls.struct<"in" { "val0" : i32, "val1" : i8, "val2" : i16 }>) -> i8 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.field<"val1"> %arg0 : <"in" { "val0" : i32, "val1" : i8, "val2" : i16 }>
  %result = spechls.field<"val1"> %in : !spechls.struct<"in" { "val0" : i32, "val1" : i8, "val2" : i16 }>
  spechls.commit %true, %result : i8
}
