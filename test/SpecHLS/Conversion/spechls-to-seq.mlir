// RUN: spechls-opt --spechls-to-seq %s | FileCheck %s
module {


  hw.module @mu_seq(out result : i32, in %arg0 : i32, in %clk : !seq.clock, in %rst : i1, in %ce : i1) {
    %true = hw.constant true
    %mu = spechls.mu<"x">(%arg0, %arg0) : i32
    hw.output %mu : i32
  }

hw.module @delay_seq(out result : i32, in %input : i32, in %clk : !seq.clock, in %rst : i1, in %ce : i1) {
    %true = hw.constant true
    %one = hw.constant 1 : i32
     %d = spechls.delay %input by 4 if %true init %one : i32
    hw.output %d : i32
  }

  hw.module @array_raw(out result : i32, in %arg0 : !spechls.array<i32, 4>, in %arg1 : i32, in %arg2 : i32, in %clk : !seq.clock, in %rst : i1, in %ce : i1) {
    %true = hw.constant true
    %mu = spechls.mu<"x">(%0, %arg0) : !spechls.array<i32, 4>
    %0 = spechls.alpha %mu[%arg1: i32], %arg2 if %true : !spechls.array<i32, 4>
    %1 = spechls.load %0[%arg1 : i32] : <i32, 4>
    hw.output %1 : i32
  }

}

