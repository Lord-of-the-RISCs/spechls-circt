// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.htask @task
spechls.htask @task(%in1 : i32, %in2 : i1) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.delay %arg0 by 1 : i32
  // CHECK: spechls.delay %0 by 2 if %arg1 : i32
  // CHECK: spechls.delay %1 by 1 init %arg0 : i32
  // CHECK: spechls.delay %2 by 4 if %arg1 init %arg0 : i32
  // CHECK: %4 = spechls.delay %2 by 4 if %arg1 init %4 : i32
  %d0 = spechls.delay %in1 by 1 : i32
  %d1 = spechls.delay %d0 by 2 if %in2 : i32
  %d2 = spechls.delay %d1 by 1 init %in1 : i32
  %d3 = spechls.delay %d2 by 4 if %in2 init %in1 : i32
  %d4 = spechls.delay %d2 by 4 if %in2 init %d4 : i32
  spechls.commit %true, %d3 : i32
}
