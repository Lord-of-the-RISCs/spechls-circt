// RUN: spechls-opt -split-input-file --remove-intra-raw %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%val: i32, %idx: i4, %array : !spechls.array<i32, 16>) -> i32 {
  %true = hw.constant 1 : i1
  %mu = spechls.mu<"arr">(%array, %next_array) : !spechls.array<i32, 16>
  %next_array = spechls.alpha %mu[%idx : i4], %val if %true : !spechls.array<i32, 16>
  // CHECK: %1 = spechls.load %mu[%arg1 : i4] : <i32, 16>
  // CHECK: %gamma = spechls.gamma<"aliasDetection">(%true, %1, %arg0) : i1, i32
  // CHECK: spechls.exit if %true with %gamma : i32
  %result = spechls.load %next_array [%idx : i4] : !spechls.array<i32, 16>
  spechls.exit if %true with %result : i32
}

//---

//CHECK-LABEL: @chainedAlphas
spechls.kernel @chainedAlphas(%val1: i32, %idx1: i32, %val2 : i32, %idx2: i32, 
      %arr: !spechls.array<i32, 16>) -> i32 {
  %true = hw.constant 1 : i1
  %mu = spechls.mu<"arr">(%arr, %next_array) : !spechls.array<i32, 16>
  %first_write = spechls.alpha %mu[%idx1 : i32], %val1 if %true : !spechls.array<i32, 16>
  %next_array = spechls.alpha %first_write[%idx2 : i32], %val2 if %true : !spechls.array<i32, 16>
  // CHECK: %2 = spechls.load %mu[%arg1 : i32] : <i32, 16>
  // CHECK: %gamma = spechls.gamma<"aliasDetection">(%true, %2, %arg0) : i1, i32
  // CHECK: %3 = comb.icmp eq %arg1, %arg3 : i32
  // CHECK: %gamma_0 = spechls.gamma<"aliasDetection">(%3, %gamma, %arg2) : i1, i32
  // CHECK: spechls.exit if %true with %gamma_0 : i32
  %result = spechls.load %next_array[%idx1 : i32] : !spechls.array<i32, 16>
  spechls.exit if %true with %result : i32
}

//---

// CHECK-LABEL: @outsideArray
spechls.kernel @outsideArray(%val: i32, %idx: i4, %arr2 : !spechls.array<i32, 16>) -> i32 {
  %true = hw.constant 1 : i1
  %next_array = spechls.alpha %arr2[%idx : i4], %val if %true : !spechls.array<i32, 16>
  // CHECK: %1 = spechls.load %arg2[%arg1 : i4] : <i32, 16>
  // CHECK: %gamma = spechls.gamma<"aliasDetection">(%true, %1, %arg0) : i1, i32
  // CHECK: spechls.exit if %true with %gamma : i32
  %result = spechls.load %next_array[%idx : i4] : !spechls.array<i32, 16>
  spechls.exit if %true with %result : i32
}

//---

// CHECK-LABEL: @throughGamma
spechls.kernel @throughGamma(%val1 : i32, %idx1 : i32, %val2 : i32, %idx2 : i32,
  %arr : !spechls.array<i32, 16>, %cond : i1, %idx_load : i32) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_array) : !spechls.array<i32, 16>
    %first_write = spechls.alpha %mu[%idx1 : i32], %val1 if %true : !spechls.array<i32, 16>
    %second_write = spechls.alpha %first_write[%idx2 : i32], %val2 if %true : !spechls.array<i32, 16>
    %next_array = spechls.gamma<"next_x">(%cond, %first_write, %second_write) : i1, !spechls.array<i32, 16>
    // CHECK: %2 = spechls.load %mu[%arg6 : i32] : <i32, 16>
    // CHECK: %3 = comb.icmp eq %arg6, %arg1 : i32
    // CHECK: %gamma_0 = spechls.gamma<"aliasDetection">(%3, %2, %arg0) : i1, i32
    // CHECK: %4 = spechls.load %mu[%arg6 : i32] : <i32, 16>
    // CHECK: %5 = comb.icmp eq %arg6, %arg1 : i32
    // CHECK: %gamma_1 = spechls.gamma<"aliasDetection">(%5, %4, %arg0) : i1, i32
    // CHECK: %6 = comb.icmp eq %arg6, %arg3 : i32
    // CHECK: %gamma_2 = spechls.gamma<"aliasDetection">(%6, %gamma_1, %arg2) : i1, i32
    // CHECK: %gamma_3 = spechls.gamma<"gamma_x">(%arg5, %gamma_0, %gamma_2) : i1, i32
    // CHECK: spechls.exit if %true with %gamma_3 : i32
    %result = spechls.load %next_array[%idx_load : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}
