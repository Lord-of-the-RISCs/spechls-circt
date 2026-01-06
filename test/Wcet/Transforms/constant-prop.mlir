//RUN: spechls-opt -split-input-file --constant-prop %s | spechls-opt | FileCheck %s

//CHECK-LABEL: simplGamma
//CHECK-NOT: spechls.gamma
//CHECK: wcet.commit %arg0
wcet.core @simplGamma(%in0 : i32, %in1 : i32) -> i32 {
    %ctrl = hw.constant 0 : i1
    %result = spechls.gamma<"res">(%ctrl, %in0, %in1) : i1, i32
    wcet.commit %result : i32
  }


//CHECK-LABEL: simplLut
//CHECK-NOT: spechls.gamma
//CHECK: wcet.commit %c4_i32
wcet.core @simplLut() -> i32 {
    %ctrl = hw.constant 0 : i1
    %result = spechls.lut %ctrl [4, 5] : (i1) -> i32
    wcet.commit %result : i32
  }
