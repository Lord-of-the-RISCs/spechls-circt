// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
spechls.kernel @kernel(%in1 : i1) -> i32 {
  //CHECK: %0 = wcet.init @x : i32
  %init = wcet.init @x : i32
  spechls.exit if %in1 with %init : i32
}
