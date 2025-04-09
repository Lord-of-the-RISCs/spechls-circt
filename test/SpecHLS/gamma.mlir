// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.htask @task
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: i32, %[[arg2:[a-zA-Z0-9]+]]: i32, %[[arg3:[a-zA-Z0-9]+]]: i32
spechls.htask @task(%in1 : i32, %in2 : i32, %in3 : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %[[g:gamma[0-9]*]] = spechls.gamma<"x">(%[[arg1]], %[[arg2]], %[[arg3]])
  %g = spechls.gamma<"x">(%in1, %in2, %in3) : i32, i32
  // CHECK: spechls.commit %true, %[[g]] : i32
  spechls.commit %true, %g : i32
}
