// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: i32, %[[arg2:[a-zA-Z0-9]+]]: i32, %[[arg3:[a-zA-Z0-9]+]]: i32
spechls.kernel @kernel(%in1 : i32, %in2 : i32, %in3 : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %[[g:gamma[0-9]*]] = spechls.gamma<"x">(%[[arg1]], %[[arg2]], %[[arg3]])
  %g = spechls.gamma<"x">(%in1, %in2, %in3) : i32, i32
  // CHECK: spechls.exit if %true with %[[g]] : i32
  spechls.exit if %true with %g : i32
}
