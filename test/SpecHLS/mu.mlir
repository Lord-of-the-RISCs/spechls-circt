// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : i32, %in2 : i32, %in3 : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %mu = spechls.mu<"x">(%arg0, %gamma)
  %m = spechls.mu<"x">(%in1, %g) : i32
  %g = spechls.gamma<"x">(%in2, %m, %in3) : i32, i32
  spechls.commit %true, %m : i32
}
