// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
spechls.kernel @kernel(%in1 : i32, %in2 : i1) -> i32 {
  //CHECK: %0:2 = wcet.dummy %arg0, %arg1 : i32, i1 -> i32, i1
  %dummy_0, %dummy_1 = wcet.dummy %in1, %in2 : i32, i1 -> i32, i1
  spechls.exit if %dummy_1 with %dummy_0 : i32
}
