// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : !spechls.struct<i32, i32, i1>) -> i32 {
  // CHECK: spechls.unpack %arg0 : <i32, i32, i1>
  %0:3 = spechls.unpack %in1 : !spechls.struct<i32, i32, i1>
  spechls.commit %0#0 : i32
}
