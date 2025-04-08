// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : !spechls.struct<i1, i32>) -> i32 {
  // CHECK: spechls.fifo<4> %arg0 : !spechls.struct<i1, i32>
  %0 = spechls.fifo<4> %in1 : !spechls.struct<i1, i32>
  spechls.commit %0 : i32
}
