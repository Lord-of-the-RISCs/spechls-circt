// RUN: spechls-opt -split-input-file --remove-intra-raw %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%val: i32, %idx: i4, %array : !spechls.array<i32, 16>, %we : i1) -> i32 {
  %true = hw.constant 1 : i1
  %mu = spechls.mu<"arr">(%array, %next_array) : !spechls.array<i32, 16>
  %next_array = spechls.alpha %mu[%idx : i4], %val if %we : !spechls.array<i32, 16>
  // CHECK: %1 = spechls.load %mu[%arg1 : i4] {spechls.scn = 0 : i32} : <i32, 16>
  // CHECK: %gamma = spechls.gamma<"aliasDetection">(%arg3, %1, %arg0) {spechls.scn = 0 : i32}: i1, i32
  %result = spechls.load %next_array [%idx : i4] : !spechls.array<i32, 16>
  spechls.exit if %true with %result : i32
}

//---

//CHECK-LABEL: @chainedAlphas
spechls.kernel @chainedAlphas(%val1: i32, %idx1: i32, %val2 : i32, %idx2: i32, 
      %arr: !spechls.array<i32, 16>, %we : i1) -> i32 {
  %true = hw.constant 1 : i1
  %mu = spechls.mu<"arr">(%arr, %next_array) : !spechls.array<i32, 16>
  %first_write = spechls.alpha %mu[%idx1 : i32], %val1 if %we : !spechls.array<i32, 16>
  %next_array = spechls.alpha %first_write[%idx2 : i32], %val2 if %we : !spechls.array<i32, 16>
  // CHECK: %2 = spechls.load %mu[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
  // CHECK: %gamma = spechls.gamma<"aliasDetection">(%arg5, %2, %arg0) {spechls.scn = 0 : i32}: i1, i32
  // CHECK: %3 = comb.icmp eq %arg1, %arg3 {spechls.scn = 0 : i32} : i32
  // CHECK: %4 = comb.and %3, %arg5 : i1
  // CHECK: %gamma_0 = spechls.gamma<"aliasDetection">(%4, %gamma, %arg2) {spechls.scn = 0 : i32}: i1, i32
  %result = spechls.load %next_array[%idx1 : i32] : !spechls.array<i32, 16>
  spechls.exit if %true with %result : i32
}

//---

// CHECK-LABEL: @outsideArray
spechls.kernel @outsideArray(%val: i32, %idx: i4, %arr2 : !spechls.array<i32, 16>, %we : i1) -> i32 {
  %true = hw.constant 1 : i1
  %next_array = spechls.alpha %arr2[%idx : i4], %val if %we : !spechls.array<i32, 16>
  // CHECK: %1 = spechls.load %arg2[%arg1 : i4] {spechls.scn = 0 : i32} : <i32, 16>
  // CHECK: %gamma = spechls.gamma<"aliasDetection">(%arg3, %1, %arg0) {spechls.scn = 0 : i32}: i1, i32
  %result = spechls.load %next_array[%idx : i4] : !spechls.array<i32, 16>
  spechls.exit if %true with %result : i32
}

//---

// CHECK-LABEL: @throughGamma
spechls.kernel @throughGamma(%val1 : i32, %idx1 : i32, %val2 : i32, %idx2 : i32,
  %arr : !spechls.array<i32, 16>, %cond : i1, %idx_load : i32, %we : i1) -> i32 {
  //CHECK: %0 = spechls.alpha %mu[%arg1: i32], %arg0 if %arg7 {spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %1 = spechls.alpha %0[%arg3: i32], %arg2 if %arg7 {spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %gamma = spechls.gamma<"next_x">(%arg5, %0, %1) {spechls.scn = 0 : i32}: i1, !spechls.array<i32, 16>
  //CHECK: %2 = spechls.load %mu[%arg6 : i32] {spechls.scn = 0 : i32} : <i32, 16>
  //CHECK: %3 = comb.icmp eq %arg6, %arg1 {spechls.scn = 0 : i32} : i32
  //CHECK: %4 = comb.and %3, %arg7 : i1
  //CHECK: %gamma_0 = spechls.gamma<"aliasDetection">(%4, %2, %arg0) {spechls.scn = 0 : i32}: i1, i32
  //CHECK: %5 = spechls.load %mu[%arg6 : i32] {spechls.scn = 0 : i32} : <i32, 16>
  //CHECK: %6 = comb.icmp eq %arg6, %arg1 {spechls.scn = 0 : i32} : i32
  //CHECK: %7 = comb.and %6, %arg7 : i1
  //CHECK: %gamma_1 = spechls.gamma<"aliasDetection">(%7, %5, %arg0) {spechls.scn = 0 : i32}: i1, i32
  //CHECK: %8 = comb.icmp eq %arg6, %arg3 {spechls.scn = 0 : i32} : i32
  //CHECK: %9 = comb.and %8, %arg7 : i1
  //CHECK: %gamma_2 = spechls.gamma<"aliasDetection">(%9, %gamma_1, %arg2) {spechls.scn = 0 : i32}: i1, i32
  //CHECK: %gamma_3 = spechls.gamma<"gamma_x">(%arg5, %gamma_0, %gamma_2) {spechls.scn = 0 : i32}: i1, i32
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_array) : !spechls.array<i32, 16>
    %first_write = spechls.alpha %mu[%idx1 : i32], %val1 if %we : !spechls.array<i32, 16>
    %second_write = spechls.alpha %first_write[%idx2 : i32], %val2 if %we : !spechls.array<i32, 16>
    %next_array = spechls.gamma<"next_x">(%cond, %first_write, %second_write) : i1, !spechls.array<i32, 16>
    %result = spechls.load %next_array[%idx_load : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}
