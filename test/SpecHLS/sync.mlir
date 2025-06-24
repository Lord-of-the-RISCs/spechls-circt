// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
spechls.kernel @kernel(%in1 : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %0 = spechls.sync %arg0, %arg0, %arg0 : i32, i32, i32
  %0 = spechls.sync %in1, %in1, %in1 : i32, i32, i32
  spechls.exit if %true with %0 : i32
}
