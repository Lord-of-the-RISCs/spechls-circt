// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : i32) -> i32 {
  // CHECK: spechls.call @encoder(%arg0) : (i32, i32) -> i32
  %0 = spechls.call @encoder(%in1, %in1) : (i32, i32) -> i32
  spechls.commit %0 : i32
}
