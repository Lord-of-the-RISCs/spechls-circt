// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
spechls.kernel @kernel(%in1 : i32, %in2 : i1) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: wcet.penalty %arg0 by 1 {} : i32
  // CHECK: wcet.penalty %0 by 2 if %arg1 {} : i32
  // CHECK: wcet.penalty %1 by 1 init %arg0 {} : i32
  // CHECK: wcet.penalty %2 by 4 if %arg1 init %arg0 {} : i32
  // CHECK: %4 = wcet.penalty %2 by 4 if %arg1 init %4 {} : i32
  %d0 = wcet.penalty %in1 by 1 : i32
  %d1 = wcet.penalty %d0 by 2 if %in2 : i32
  %d2 = wcet.penalty %d1 by 1 init %in1 : i32
  %d3 = wcet.penalty %d2 by 4 if %in2 init %in1 : i32
  %d4 = wcet.penalty %d2 by 4 if %in2 init %d4 : i32
  spechls.exit if %in2 with %d4 : i32
}
