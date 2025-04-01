// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.htask @task
spechls.htask @task(%in1 : i32) -> i3 {
  // CHECK: %[[l:lut[0-9]*]] = spechls.lut %arg0 [0, 1, 2, 3, 4, 5, 6, 7] : (i32) -> i3
  %l = spechls.lut %in1 [0, 1, 2, 3, 4, 5, 6, 7] : (i32) -> i3
  spechls.commit %l : i3
}
