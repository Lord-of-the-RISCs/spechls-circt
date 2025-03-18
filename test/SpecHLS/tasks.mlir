// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @toplevel
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: i32, %[[arg2:[a-zA-Z0-9]+]]: i1
spechls.hkernel @toplevel(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %[[res:.+]] = spechls.launch @task1(%[[arg1]]) : (i32) -> i32
  %0 = spechls.launch @task1(%in1) : (i32) -> i32
  // CHECK: %[[val:.+]] = spechls.launch @task2(%[[res]]) : (i32) -> i32
  %1 = spechls.launch @task2(%0) : (i32) -> i32
  // CHECK: spechls.exit if %[[arg2]] with %[[val]] : i32
  spechls.exit if %in2 with %1 : i32
}

// CHECK: spechls.htask @task1
// CHECK-SAME: %[[arg:[a-zA-Z0-9]+]]
spechls.htask @task1(%in1 : i32) -> i32 {
  // CHECK: spechls.commit %[[arg]] : i32
  spechls.commit %in1 : i32
}

spechls.htask @task2(%in1 : i32) -> i32 {
  spechls.commit %in1 : i32
}
