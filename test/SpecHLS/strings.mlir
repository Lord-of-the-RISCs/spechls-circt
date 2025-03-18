// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%in1 : i32, %in2 : i32, %in3 : i32) -> !spechls.str {
  // CHECK: %str = spechls.string "Hello, world!"
  %str = spechls.string "Hello, world!"
  spechls.commit %str : !spechls.str
}
