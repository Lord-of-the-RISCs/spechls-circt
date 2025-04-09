// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : !spechls.struct<i32, i16, i1>) -> !spechls.struct<i1, i16, i32> {
  // CHECK: spechls.unpack %arg0 : <i32, i16, i1>
  // CHECK: spechls.pack %0#2, %0#1, %0#0 : i1, i16, i32
  %0:3 = spechls.unpack %in1 : !spechls.struct<i32, i16, i1>
  %1 = spechls.pack %0#2, %0#1, %0#0 : i1, i16, i32
  spechls.commit %1 : !spechls.struct<i1, i16, i32>
}
