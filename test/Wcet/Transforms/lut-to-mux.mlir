//RUN: spechls-opt -split-input-file --LUT-to-mux %s | spechls-opt | FileCheck %s

//CHECK-LABEL: @simple
//CHECK-NOT: spechls.lut
//CHECK: comb.mux %1, %c2_i32, %c0_i32
//CHECK: comb.mux %1, %c3_i32, %c1_i32
//CHECK: comb.mux %0, %3, %2
spechls.kernel @simple(%ctrl : i2) -> i32 {
    %true = hw.constant 1 : i1
    %result = spechls.lut %ctrl [0,1,2,3] : (i2) -> i32
    spechls.exit if %true with %result : i32
  }
