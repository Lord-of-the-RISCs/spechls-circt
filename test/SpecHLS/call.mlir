// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel(%in1 : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: spechls.call @encoder(%arg0, %arg0) : (i32, i32) -> i32
  %0 = spechls.call @encoder(%in1, %in1) : (i32, i32) -> i32
  spechls.exit if %true with %0 : i32
}
