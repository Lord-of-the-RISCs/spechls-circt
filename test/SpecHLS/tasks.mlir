// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @toplevel
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: i32, %[[arg2:[a-zA-Z0-9]+]]: i1
spechls.kernel @toplevel(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %[[res:.+]] = spechls.launch @task1(%[[arg1]]) : (i32) -> i32
  %0 = spechls.launch @task1(%in1) : (i32) -> i32
  // CHECK: %[[val:.+]] = spechls.launch @task2(%[[res]]) : (i32) -> i32
  %1 = spechls.launch @task2(%0) : (i32) -> i32
  // CHECK: %2:2 = spechls.launch @task3(%[[res]]) : (i32) -> (i32, i32)
  %2, %3 = spechls.launch @task3(%0) : (i32) -> (i32, i32)
  // CHECK: spechls.exit if %[[arg2]]
  spechls.exit if %in2
}

// CHECK: spechls.task @task1
// CHECK-SAME: %[[arg:[a-zA-Z0-9]+]]
spechls.task @task1(%in1 : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.commit %true, %[[arg]] : i32
  spechls.commit %true, %in1 : i32
}
