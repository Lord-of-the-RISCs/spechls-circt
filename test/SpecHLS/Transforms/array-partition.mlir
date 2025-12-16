// RUN: spechls-opt -split-input-file  --expose-memory-speculation %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%arr : !spechls.array<i32,256>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) : !spechls.array<i32,256>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32,256>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32,256>
    spechls.exit if %true with %result : i32
}
