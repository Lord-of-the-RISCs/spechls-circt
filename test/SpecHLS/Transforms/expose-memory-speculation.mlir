// RUN: spechls-opt -split-input-file --remove-intra-raw --expose-memory-speculation %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%arr : !spechls.array<i32, 16>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16> 
    //CHECK: %0 = spechls.rollbackableDelay %8 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = spechls.delay %arg2 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %3 = spechls.cancellableDelay %arg4 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %4 = comb.icmp eq %2, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %5 = comb.and %4, %3 {spechls.scn = 0 : i32} : i1
    //CHECK: %6 = comb.mux %5, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %7 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%6, %0, %7) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %8) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %8 = spechls.alpha %mu[%arg2: i32], %arg3 if %arg4 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @simpleD3
spechls.kernel @simpleD3(%arr : !spechls.array<i32, 16>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    //CHECK: %0 = spechls.rollbackableDelay %14 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = spechls.delay %arg2 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %3 = spechls.delay %arg2 by 2 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %4 = spechls.cancellableDelay %arg4 by 1 cancel %false at 2 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %5 = spechls.cancellableDelay %arg4 by 2 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %6 = comb.icmp eq %3, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %7 = comb.and %6, %5 {spechls.scn = 0 : i32} : i1
    //CHECK: %8 = comb.mux %7, %c1_i32, %c2_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %9 = comb.icmp eq %2, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %10 = comb.and %9, %4 {spechls.scn = 0 : i32} : i1
    //CHECK: %11 = comb.mux %10, %c0_i32, %8 {spechls.scn = 0 : i32} : i32
    //CHECK: %12 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %13 = spechls.rollbackableDelay %12 by 1 rollback %false at 2 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%11, %0, %12, %13) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %14) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %14 = spechls.alpha %mu[%arg2: i32], %arg3 if %arg4 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 3} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @twoRead
spechls.kernel @twoRead(%arr : !spechls.array<i32, 16>, %idxRead1 : i32, %idxRead2 : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    //CHECK: %0 = spechls.rollbackableDelay %12 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg2 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = spechls.load %gamma[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %3 = spechls.delay %arg3 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %4 = spechls.cancellableDelay %arg5 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %5 = comb.icmp eq %3, %arg2 {spechls.scn = 0 : i32} : i32
    //CHECK: %6 = comb.and %5, %4 {spechls.scn = 0 : i32} : i1
    //CHECK: %7 = comb.icmp eq %3, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %8 = comb.and %7, %4 {spechls.scn = 0 : i32} : i1
    //CHECK: %9 = comb.or %8, %6 {spechls.scn = 0 : i32} : i1
    //CHECK: %10 = comb.mux %9, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %11 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%10, %0, %11) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %12) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %12 = spechls.alpha %mu[%arg3: i32], %arg4 if %arg5 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    %operand1 = spechls.load %mu[%idxRead1 : i32] : !spechls.array<i32, 16>
    %operand2 = spechls.load %mu[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @twoWrite
spechls.kernel @twoWrite(
    %arr : !spechls.array<i32, 16>, %idxRead : i32,
    %idxWrite1 : i32, %idxWrite2 : i32,
    %valWrite1 : i32, %valWrite2 : i32,
    %we1 : i1, %we2 : i1) -> i32 
{
    //CHECK: %0 = spechls.rollbackableDelay %14 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = spechls.delay %arg2 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %3 = spechls.delay %arg3 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %4 = spechls.cancellableDelay %arg6 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %5 = spechls.cancellableDelay %arg7 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %6 = comb.icmp eq %2, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %7 = comb.and %6, %4 {spechls.scn = 0 : i32} : i1
    //CHECK: %8 = comb.icmp eq %3, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %9 = comb.and %8, %5 {spechls.scn = 0 : i32} : i1
    //CHECK: %10 = comb.or %7, %9 {spechls.scn = 0 : i32} : i1
    //CHECK: %11 = comb.mux %10, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %12 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%11, %0, %12) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %14) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %13 = spechls.alpha %mu[%arg2: i32], %arg4 if %arg6 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %14 = spechls.alpha %13[%arg3: i32], %arg5 if %arg7 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %temp_arr = spechls.alpha %mu[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %temp_arr[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}


// CHECK-LABEL: @twoReadsTwoWritesD3
spechls.kernel @twoReadsTwoWritesD3(
    %arr : !spechls.array<i32, 16>,
    %idxRead1 : i32, %idxRead2 : i32,
    %idxWrite1 : i32, %idxWrite2 : i32,
    %valWrite1 : i32, %valWrite2 : i32,
    %we1 : i1, %we2 : i1) -> i32 
{
    //CHECK: %0 = spechls.rollbackableDelay %38 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg2 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = spechls.load %gamma[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %3 = spechls.delay %arg3 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %4 = spechls.delay %arg4 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %5 = spechls.delay %arg3 by 2 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %6 = spechls.delay %arg4 by 2 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %7 = spechls.cancellableDelay %arg7 by 1 cancel %false at 2 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %8 = spechls.cancellableDelay %arg8 by 1 cancel %false at 2 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %9 = spechls.cancellableDelay %arg7 by 2 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %10 = spechls.cancellableDelay %arg8 by 2 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %11 = comb.icmp eq %5, %arg2 {spechls.scn = 0 : i32} : i32
    //CHECK: %12 = comb.and %11, %9 {spechls.scn = 0 : i32} : i1
    //CHECK: %13 = comb.icmp eq %6, %arg2 {spechls.scn = 0 : i32} : i32
    //CHECK: %14 = comb.and %13, %10 {spechls.scn = 0 : i32} : i1
    //CHECK: %15 = comb.or %12, %14 {spechls.scn = 0 : i32} : i1
    //CHECK: %16 = comb.icmp eq %5, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %17 = comb.and %16, %9 {spechls.scn = 0 : i32} : i1
    //CHECK: %18 = comb.icmp eq %6, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %19 = comb.and %18, %10 {spechls.scn = 0 : i32} : i1
    //CHECK: %20 = comb.or %17, %19 {spechls.scn = 0 : i32} : i1
    //CHECK: %21 = comb.or %20, %15 {spechls.scn = 0 : i32} : i1
    //CHECK: %22 = comb.mux %21, %c1_i32, %c2_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %23 = comb.icmp eq %3, %arg2 {spechls.scn = 0 : i32} : i32
    //CHECK: %24 = comb.and %23, %7 {spechls.scn = 0 : i32} : i1
    //CHECK: %25 = comb.icmp eq %4, %arg2 {spechls.scn = 0 : i32} : i32
    //CHECK: %26 = comb.and %25, %8 {spechls.scn = 0 : i32} : i1
    //CHECK: %27 = comb.or %24, %26 {spechls.scn = 0 : i32} : i1
    //CHECK: %28 = comb.icmp eq %3, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %29 = comb.and %28, %7 {spechls.scn = 0 : i32} : i1
    //CHECK: %30 = comb.icmp eq %4, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %31 = comb.and %30, %8 {spechls.scn = 0 : i32} : i1
    //CHECK: %32 = comb.or %29, %31 {spechls.scn = 0 : i32} : i1
    //CHECK: %33 = comb.or %32, %27 {spechls.scn = 0 : i32} : i1
    //CHECK: %34 = comb.mux %33, %c0_i32, %22 {spechls.scn = 0 : i32} : i32
    //CHECK: %35 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %36 = spechls.rollbackableDelay %35 by 1 rollback %false at 2 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%34, %0, %35, %36) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %38) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %37 = spechls.alpha %mu[%arg3: i32], %arg5 if %arg7 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %38 = spechls.alpha %37[%arg4: i32], %arg6 if %arg8 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 3} : !spechls.array<i32, 16>
    %temp_arr = spechls.alpha %mu[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %temp_arr[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    %operand1 = spechls.load %mu[%idxRead1 : i32] : !spechls.array<i32, 16>
    %operand2 = spechls.load %mu[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @twoMemSpecD3D2
spechls.kernel @twoMemSpecD3D2(
    %arr1 : !spechls.array<i32, 16>,
    %arr2 : !spechls.array<i32, 16>,
    %idxRead1 : i32, %idxRead2 : i32,
    %idxWrite1 : i32, %idxWrite2 : i32,
    %valWrite1 : i32, %valWrite2 : i32,
    %we1 : i1, %we2 : i1) -> i32 
{
  //CHECK: %0 = spechls.rollbackableDelay %22 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 1 : i32} : !spechls.array<i32, 16>
  //CHECK: %1 = spechls.load %gamma[%arg2 : i32] {spechls.scn = 1 : i32} : <i32, 16>
  //CHECK: %2 = spechls.delay %arg4 by 1 if %true {spechls.scn = 1 : i32} : i32
  //CHECK: %3 = spechls.delay %arg4 by 2 if %true {spechls.scn = 1 : i32} : i32
  //CHECK: %4 = spechls.cancellableDelay %arg8 by 1 cancel %false at 2 if %true init %false {spechls.scn = 1 : i32} : i1
  //CHECK: %5 = spechls.cancellableDelay %arg8 by 2 cancel %false at 1 if %true init %false {spechls.scn = 1 : i32} : i1
  //CHECK: %6 = comb.icmp eq %3, %arg2 {spechls.scn = 1 : i32} : i32
  //CHECK: %7 = comb.and %6, %5 {spechls.scn = 1 : i32} : i1
  //CHECK: %8 = comb.mux %7, %c1_i32, %c2_i32 {spechls.scn = 1 : i32} : i32
  //CHECK: %9 = comb.icmp eq %2, %arg2 {spechls.scn = 1 : i32} : i32
  //CHECK: %10 = comb.and %9, %4 {spechls.scn = 1 : i32} : i1
  //CHECK: %11 = comb.mux %10, %c0_i32, %8 {spechls.scn = 1 : i32} : i32
  //CHECK: %12 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 1 : i32} : !spechls.array<i32, 16>
  //CHECK: %13 = spechls.rollbackableDelay %12 by 1 rollback %false at 2 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 1 : i32} : !spechls.array<i32, 16>
  //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%11, %0, %12, %13) {spechls.memspec, spechls.scn = 1 : i32}: i32, !spechls.array<i32, 16>
  //CHECK: %mu = spechls.mu<"x">(%arg0, %22) {spechls.scn = 1 : i32}: !spechls.array<i32, 16>
  //CHECK: %14 = spechls.rollbackableDelay %23 by 1 rollback %false at 0 %false [] if %true init %arg1 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %15 = spechls.load %gamma_0[%arg3 : i32] {spechls.scn = 0 : i32} : <i32, 16>
  //CHECK: %16 = spechls.delay %arg5 by 1 if %true {spechls.scn = 0 : i32} : i32
  //CHECK: %17 = spechls.cancellableDelay %arg9 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
  //CHECK: %18 = comb.icmp eq %16, %arg3 {spechls.scn = 0 : i32} : i32
  //CHECK: %19 = comb.and %18, %17 {spechls.scn = 0 : i32} : i1
  //CHECK: %20 = comb.mux %19, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
  //CHECK: %21 = spechls.rollbackableDelay %14 by 1 rollback %false at 1 %false [] if %true init %arg1 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %gamma_0 = spechls.gamma<"memory_speculation_y">(%20, %14, %21) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
  //CHECK: %mu_1 = spechls.mu<"y">(%arg1, %23) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
  //CHECK: %22 = spechls.alpha %mu[%arg4: i32], %arg6 if %arg8 {spechls.memspec, spechls.scn = 1 : i32} : !spechls.array<i32, 16>
  //CHECK: %23 = spechls.alpha %mu_1[%arg5: i32], %arg7 if %arg9 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %true = hw.constant 1 : i1
    %mu1 = spechls.mu<"x">(%arr1, %next_arr1) {spechls.memspec = 3} : !spechls.array<i32, 16>
    %mu2 = spechls.mu<"y">(%arr2, %next_arr2) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %next_arr1 = spechls.alpha %mu1[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr2 = spechls.alpha %mu2[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    %operand1 = spechls.load %mu1[%idxRead1 : i32] : !spechls.array<i32, 16>
    %operand2 = spechls.load %mu2[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @simpleWithDiffType
spechls.kernel @simpleWithDiffType(%arr : !spechls.array<i32, 16>, %idxRead : si32, %idxWrite : si32, %valWrite : i32, %we : i1) -> i32 {
    //CHECK: %0 = spechls.rollbackableDelay %10 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg1 : si32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = hw.bitcast %arg1 {spechls.scn = 0 : i32} : (si32) -> i32
    //CHECK: %3 = hw.bitcast %arg2 {spechls.scn = 0 : i32} : (si32) -> i32
    //CHECK: %4 = spechls.delay %3 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %5 = spechls.cancellableDelay %arg4 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %6 = comb.icmp eq %4, %2 {spechls.scn = 0 : i32} : i32
    //CHECK: %7 = comb.and %6, %5 {spechls.scn = 0 : i32} : i1
    //CHECK: %8 = comb.mux %7, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %9 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%8, %0, %9) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %10) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %10 = spechls.alpha %mu[%arg2: si32], %arg3 if %arg4 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : si32], %valWrite if %we : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : si32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

spechls.kernel @simpleDoubleRead(%arr : !spechls.array<i32, 16>, %idxRead1 : i32, %idxRead2 : i32,
    //CHECK: %0 = spechls.rollbackableDelay %12 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg1 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = spechls.load %gamma[%arg2 : i32] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %3 = spechls.delay %arg3 by 1 if %true {spechls.scn = 0 : i32} : i32
    //CHECK: %4 = spechls.cancellableDelay %arg5 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %5 = comb.icmp eq %3, %arg1 {spechls.scn = 0 : i32} : i32
    //CHECK: %6 = comb.and %5, %4 {spechls.scn = 0 : i32} : i1
    //CHECK: %7 = comb.icmp eq %3, %arg2 {spechls.scn = 0 : i32} : i32
    //CHECK: %8 = comb.and %7, %4 {spechls.scn = 0 : i32} : i1
    //CHECK: %9 = comb.or %8, %6 {spechls.scn = 0 : i32} : i1
    //CHECK: %10 = comb.mux %9, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %11 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%10, %0, %11) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %12) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %12 = spechls.alpha %mu[%arg3: i32], %arg4 if %arg5 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %13 = comb.icmp eq %arg1, %arg3 {spechls.scn = 0 : i32} : i32
    //CHECK: %14 = comb.and %13, %arg5 : i1
    //CHECK: %gamma_0 = spechls.gamma<"aliasDetection">(%14, %1, %arg4) {spechls.scn = 0 : i32}: i1, i32
    //CHECK: %15 = comb.icmp eq %arg2, %arg3 {spechls.scn = 0 : i32} : i32
    //CHECK: %16 = comb.and %15, %arg5 : i1
    //CHECK: %gamma_1 = spechls.gamma<"aliasDetection">(%16, %2, %arg4) {spechls.scn = 0 : i32}: i1, i32
    %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    %operand1 = spechls.load %next_arr[%idxRead1 : i32] : !spechls.array<i32, 16>
    %operand2 = spechls.load %next_arr[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}

spechls.kernel @simpleUnsigned(%arr : !spechls.array<i32, 16>, %idxRead : ui8,
    //CHECK: %0 = spechls.rollbackableDelay %10 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %1 = spechls.load %gamma[%arg1 : ui8] {spechls.scn = 0 : i32} : <i32, 16>
    //CHECK: %2 = hw.bitcast %arg1 {spechls.scn = 0 : i32} : (ui8) -> i8
    //CHECK: %3 = hw.bitcast %arg2 {spechls.scn = 0 : i32} : (ui8) -> i8
    //CHECK: %4 = spechls.delay %3 by 1 if %true {spechls.scn = 0 : i32} : i8
    //CHECK: %5 = spechls.cancellableDelay %arg4 by 1 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
    //CHECK: %6 = comb.icmp eq %4, %2 {spechls.scn = 0 : i32} : i8
    //CHECK: %7 = comb.and %6, %5 {spechls.scn = 0 : i32} : i1
    //CHECK: %8 = comb.mux %7, %c0_i32, %c1_i32 {spechls.scn = 0 : i32} : i32
    //CHECK: %9 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    //CHECK: %gamma = spechls.gamma<"memory_speculation_x">(%8, %0, %9) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
    //CHECK: %mu = spechls.mu<"x">(%arg0, %10) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
    //CHECK: %10 = spechls.alpha %mu[%arg2: ui8], %arg3 if %arg4 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
    %idxWrite : ui8, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {spechls.memspec = 2} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : ui8], %valWrite if %we : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : ui8] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}


spechls.kernel @withFields(%arr : !spechls.array<i32, 16>, %in : !spechls.struct<"in" { "val" : i32,  "wAddr" : ui8, "we" : i1, 
"rdAddr" : ui8 }>) -> i32 {
  //CHECK: %0 = spechls.rollbackableDelay %19 by 1 rollback %false at 0 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %1 = spechls.load %gamma[%20 : ui8] {spechls.scn = 0 : i32} : <i32, 16>
  //CHECK: %2 = hw.bitcast %20 {spechls.scn = 0 : i32} : (ui8) -> i8
  //CHECK: %3 = hw.bitcast %16 {spechls.scn = 0 : i32} : (ui8) -> i8
  //CHECK: %4 = spechls.delay %3 by 1 if %true {spechls.scn = 0 : i32} : i8
  //CHECK: %5 = spechls.delay %3 by 2 if %true {spechls.scn = 0 : i32} : i8
  //CHECK: %6 = spechls.cancellableDelay %17 by 1 cancel %false at 2 if %true init %false {spechls.scn = 0 : i32} : i1
  //CHECK: %7 = spechls.cancellableDelay %17 by 2 cancel %false at 1 if %true init %false {spechls.scn = 0 : i32} : i1
  //CHECK: %8 = comb.icmp eq %5, %2 {spechls.scn = 0 : i32} : i8
  //CHECK: %9 = comb.and %8, %7 {spechls.scn = 0 : i32} : i1
  //CHECK: %10 = comb.mux %9, %c1_i32, %c2_i32 {spechls.scn = 0 : i32} : i32
  //CHECK: %11 = comb.icmp eq %4, %2 {spechls.scn = 0 : i32} : i8
  //CHECK: %12 = comb.and %11, %6 {spechls.scn = 0 : i32} : i1
  //CHECK: %13 = comb.mux %12, %c0_i32, %10 {spechls.scn = 0 : i32} : i32
  //CHECK: %14 = spechls.rollbackableDelay %0 by 1 rollback %false at 1 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %15 = spechls.rollbackableDelay %14 by 1 rollback %false at 2 %false [] if %true init %arg0 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %gamma = spechls.gamma<"memory_speculation_arr">(%13, %0, %14, %15) {spechls.memspec, spechls.scn = 0 : i32}: i32, !spechls.array<i32, 16>
  //CHECK: %mu = spechls.mu<"arr">(%arg0, %19) {spechls.scn = 0 : i32}: !spechls.array<i32, 16>
  //CHECK: %16 = spechls.field<"wAddr"> %arg1 : <"in" { "val" : i32, "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  //CHECK: %17 = spechls.field<"we"> %arg1 : <"in" { "val" : i32, "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  //CHECK: %18 = spechls.field<"val"> %arg1 : <"in" { "val" : i32, "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  //CHECK: %19 = spechls.alpha %mu[%16: ui8], %18 if %17 {spechls.memspec, spechls.scn = 0 : i32} : !spechls.array<i32, 16>
  //CHECK: %20 = spechls.field<"rdAddr"> %arg1 : <"in" { "val" : i32, "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  //CHECK: %21 = hw.bitcast %20 {spechls.scn = 0 : i32} : (ui8) -> i8
  //CHECK: %22 = hw.bitcast %16 {spechls.scn = 0 : i32} : (ui8) -> i8
  //CHECK: %23 = comb.icmp eq %21, %22 {spechls.scn = 0 : i32} : i8
  //CHECK: %24 = comb.and %23, %17 : i1
  //CHECK: %gamma_0 = spechls.gamma<"aliasDetection">(%24, %1, %18) {spechls.scn = 0 : i32}: i1, i32
  %true = hw.constant 1 : i1
  %mu = spechls.mu<"arr">(%arr, %next_arr) {spechls.memspec=3}: !spechls.array<i32, 16>
  %wrAddr = spechls.field<"wAddr"> %in : !spechls.struct<"in" { "val" : i32,  "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  %we = spechls.field<"we"> %in : !spechls.struct<"in" { "val" : i32,  "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  %val = spechls.field<"val"> %in : !spechls.struct<"in" { "val" : i32,  "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  %next_arr = spechls.alpha %mu[%wrAddr : ui8], %val if %we : !spechls.array<i32, 16>
  %rdAddr = spechls.field<"rdAddr"> %in : !spechls.struct<"in" { "val" : i32,  "wAddr" : ui8, "we" : i1, "rdAddr" : ui8 }>
  %result = spechls.load %next_arr[%rdAddr : ui8] : <i32, 16>
  spechls.exit if %true with %result : i32
}
