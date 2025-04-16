// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in : !spechls.struct<i32, i8, i16>) -> i8 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.field<1> %arg0 : <i32, i8, i16>
  %result = spechls.field<1> %in : !spechls.struct<i32, i8, i16>
  spechls.commit %true, %result : i8
}
