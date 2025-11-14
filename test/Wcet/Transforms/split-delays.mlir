// RUN: spechls-opt -split-input-file --split-delays %s | spechls-opt | FileCheck %s

spechls.kernel @simple(%in1 : i32, %in2 : i1) -> i32 {
  //CHECK: %0 = spechls.delay %arg0 by 1 if %true {} : i32
  //CHECK: %1 = spechls.delay %0 by 1 if %true {} : i32
  //CHECK: %2 = spechls.delay %1 by 1 if %true {} : i32
  %true = hw.constant 1 : i1
  %d0 = spechls.delay %in1 by 3 if %true : i32
  spechls.exit if %in2 with %d0 : i32
}

spechls.kernel @doubleUse(%in1 : i32, %in2 : i1) -> i32 {
  //CHECK: %0 = spechls.delay %arg0 by 1 if %true {} : i32
  //CHECK: %1 = spechls.delay %0 by 1 if %true {} : i32
  //CHECK: %2 = spechls.delay %1 by 1 if %true {} : i32
  //CHECK: %3 = spechls.delay %2 by 1 {} : i32
  //CHECK: %4 = spechls.delay %3 by 1 {} : i32
  %true = hw.constant 1 : i1
  %d0 = spechls.delay %in1 by 3 if %true : i32
  %d1 = spechls.delay %d0 by 2 : i32
  %result = comb.add %d0, %d1 : i32
  spechls.exit if %in2 with %result : i32
}
