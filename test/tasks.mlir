// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @toplevel
// CHECK-SAME: %[[arg:[a-zA-Z0-9]+]]
spechls.hkernel @toplevel(%in1 : i32) -> i32 {
  // CHECK: %[[val:.+]] = spechls.launch @task(%[[arg]]) : (i32) -> i32
  %0 = spechls.launch @task(%in1) : (i32) -> i32
  // CHECK: spechls.exit %[[val]] : i32
  spechls.exit %0 : i32
}

// CHECK: spechls.htask @task
// CHECK-SAME: %[[arg:[a-zA-Z0-9]+]]
spechls.htask @task(%in1 : i32) -> i32 {
  // CHECK: spechls.commit %[[arg]] : i32
  spechls.commit %in1 : i32
}
