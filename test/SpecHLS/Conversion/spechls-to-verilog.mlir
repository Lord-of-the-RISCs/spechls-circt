// RUN: ../../prefix/bin/circt-opt    -lower-seq-hlmem   --lower-seq-to-sv   -hw-cleanup   -canonicalize  --export-split-verilogs %s | FileCheck %s
module {
  hw.module @mu_seq(out result : i32, in %arg0 : i32, in %clk : !seq.clock, in %rst : i1, in %ce : i1) {
    %true = hw.constant true
    %0 = comb.mux %rst, %arg0, %arg0 : i32
    %mu_x = seq.compreg.ce sym @mu_x %0, %clk, %ce : i32
    hw.output %mu_x : i32
  }
  hw.module @delay_seq(out result : i32, in %input : i32, in %clk : !seq.clock, in %rst : i1, in %ce : i1) {
    %true = hw.constant true
    %c1_i32 = hw.constant 1 : i32
    %d_0 = seq.compreg.ce sym @d_0 %input, %clk, %ce : i32
    %d_1 = seq.compreg.ce sym @d_1 %d_0, %clk, %ce : i32
    %d_2 = seq.compreg.ce sym @d_2 %d_1, %clk, %ce : i32
    %d_3 = seq.compreg.ce sym @d_3 %d_2, %clk, %ce : i32
    hw.output %d_3 : i32
  }
  hw.module @array_raw(out result : i32,in %arg1 : i32, in %arg2 : i32, in %clk : !seq.clock, in %rst : i1, in %ce : i1) {
    %x = seq.hlmem @x %clk, %rst : <4xi32>
    %true = hw.constant true
    %0 = comb.extract %arg1 from 0 : (i32) -> i2
    seq.write %x[%0] %arg2 wren %true {latency = 1 : i64} : !seq.hlmem<4xi32>
    %1 = comb.extract %arg1 from 0 : (i32) -> i2
    %true_0 = hw.constant true
    %x_rdata = seq.read %x[%1] rden %true_0 {latency = 0 : i64} : !seq.hlmem<4xi32>
    hw.output %x_rdata : i32
  }
}
