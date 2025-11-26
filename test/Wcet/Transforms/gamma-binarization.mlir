//RUN: spechls-opt -split-input-file --gamma-binarization %s | spechls-opt | FileCheck %s

//CHECK-LABEL: @simple
//CHECK-NOT: comb.mux
//CHECK: spechls.gamma<"gammaBin">(%arg0, %arg1, %arg2)
spechls.kernel @simple(%ctrl : i1, %in0 : i32, %in1 : i32) -> i32 {
    %C0 = hw.constant 0 : i32
    %C1 = hw.constant 1 : i32
    %true = hw.constant 1 : i1
    %gammaCtrl = comb.mux %ctrl, %C1, %C0 : i32
    %result = spechls.gamma<"x">(%gammaCtrl, %in0, %in1) : i32, i32
    spechls.exit if %true with %result : i32
}

//CHECK-LABEL: @notMuxTree
//CHECK-NOT: spechls.gamma<"gammaBin">
//CHECK: comb.mux
spechls.kernel @notMuxTree(%ctrl : i1, %in0 : i32, %in1 : i32, %in2  : i32, %in3 : i2) -> i32 {
  %true = hw.constant 1 : i1
  %c0 = hw.constant 0 : i2
  %gammaCtrl = comb.mux %ctrl, %c0, %in3 : i2
  %result = spechls.gamma<"x">(%gammaCtrl, %in0, %in1, %in2) : i2, i32
  spechls.exit if %true with %result : i32
}

//CHECK-LABEL: @muxCycle
//CHECK-NOT: spechls.gamma<"gammaBin">
//CHECK: comb.mux
spechls.kernel @muxCycle(%ctrl1 : i1, %ctrl2 : i1, %in0 : i32, %in1 : i32) -> i32 {
  %true = hw.constant 1 : i1
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %mux1 = comb.mux %ctrl1, %c0, %mux2 : i1
  %mux2 = comb.mux %ctrl2, %mux1, %c1 : i1
  %result = spechls.gamma<"x">(%mux2, %in0, %in1) : i1, i32
  spechls.exit if %true with %result : i32
}

//CHECK-LABEL: @muxTreeDepth2
//CHECK-NOT: comb.mux
//CHECK: spechls.gamma<"gammaBin">(%arg1
//CHECK: spechls.gamma<"gammaBin">(%arg0
//CHECK: spechls.gamma<"gammaBin">(%arg2
spechls.kernel @muxTreeDepth2(%ctrl0 : i1, %ctrl1 : i1, %ctrl2 : i1,
  %in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32) -> i32 {
  %true = hw.constant 1 : i1
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %mux1 = comb.mux %ctrl0, %c0, %c1 : i2
  %mux2 = comb.mux %ctrl1, %c3, %c2 : i2
  %mux3 = comb.mux %ctrl2, %mux1, %mux2 : i2
  %result = spechls.gamma<"x">(%mux3, %in0, %in1, %in2, %in3) : i2, i32
  spechls.exit if %true with %result : i32
}

//CHECK-LABEL: @muxGraph
//CHECK-NOT: comb.mux
//CHECK: spechls.gamma<"gammaBin">(%arg0
//CHECK: spechls.gamma<"gammaBin">(%arg2
//CHECK: spechls.gamma<"gammaBin">(%arg1
//CHECK: spechls.gamma<"gammaBin">(%arg3
spechls.kernel @muxGraph(%ctrl0 : i1, %ctrl1 : i1, %ctrl2 : i1, %ctrl3 : i1,
  %in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32) -> i32 {
  %true = hw.constant 1 : i1
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %mux1 = comb.mux %ctrl0, %c0, %c1 : i2
  %mux2 = comb.mux %ctrl1, %mux1, %c2 : i2
  %mux3 = comb.mux %ctrl2, %mux1, %c3 : i2
  %mux4 = comb.mux %ctrl3, %mux2, %mux3 : i2
  %result = spechls.gamma<"x">(%mux4, %in0, %in1, %in2, %in3) : i2, i32
  spechls.exit if %true with %result : i32
}
