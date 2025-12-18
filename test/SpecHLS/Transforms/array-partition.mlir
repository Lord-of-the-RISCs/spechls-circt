// RUN: spechls-opt  --partition-array="array-name=x nb-blocks=2 block-size=128"  %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%arr : !spechls.array<i32,256>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) : !spechls.array<i32,256>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32,256>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32,256>
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @simple
spechls.kernel @oddeven_read(%arr : !spechls.array<i32,256>, %i : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) : !spechls.array<i32,256>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32,256>
    %c0 = hw.constant 0 : i32
    %c1 = hw.constant 1 : i32
    %c2 = hw.constant 2 : i32
    %muli = comb.mul %i, %c2 : i32
    %idxRead0 = comb.add %muli, %c0 : i32
    %idxRead1 = comb.add %muli, %c1 : i32
    %result0 = spechls.load %mu[%idxRead0 : i32] : !spechls.array<i32,256>
       %result1 = spechls.load %mu[%idxRead1 : i32] : !spechls.array<i32,256>
       %result = comb.add  %result0, %result1 : i32
          spechls.exit if %true with %result : i32
}
