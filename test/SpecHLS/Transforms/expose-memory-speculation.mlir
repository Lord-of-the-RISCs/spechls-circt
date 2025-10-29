// RUN: spechls-opt -split-input-file  --expose-memory-speculation %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%arr : !spechls.array<i32, 16>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 1} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    //CHECK: %0 = spechls.delay %arg2 by 1 if %true init %arg2 : i32 
    //CHECK: %1 = spechls.delay %arg4 by 1 if %true init %arg4 : i1 
    //CHECK: %2 = comb.icmp eq %0, %arg1 : i32
    //CHECK: %3 = comb.and %2, %1 : i1
    //CHECK: %gamma = spechls.gamma<"alias_check_x_distance_1">(%3, %c1_i32, %c0_i32) {}: i1, i32
    //CHECK: %4 = spechls.delay %mu by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %gamma_0 = spechls.gamma<"memory_speculation_x">(%gamma, %mu, %4) {}: i32, !spechls.array<i32, 16>
    //CHECK: %5 = spechls.alpha %mu[%arg2: i32], %arg3 if %arg4 : !spechls.array<i32, 16>
    //CHECK: %6 = spechls.load %gamma_0[%arg1 : i32] : <i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @simpleD3
spechls.kernel @simpleD3(%arr : !spechls.array<i32, 16>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 3} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    //CHECK: %0 = spechls.delay %arg2 by 1 if %true init %arg2 : i32 
    //CHECK: %1 = spechls.delay %0 by 1 if %true init %0 : i32 
    //CHECK: %2 = spechls.delay %1 by 1 if %true init %1 : i32 
    //CHECK: %3 = spechls.delay %arg4 by 1 if %true init %arg4 : i1 
    //CHECK: %4 = spechls.delay %3 by 1 if %true init %3 : i1 
    //CHECK: %5 = spechls.delay %4 by 1 if %true init %4 : i1 
    //CHECK: %6 = comb.icmp eq %2, %arg1 : i32
    //CHECK: %7 = comb.and %6, %5 : i1
    //CHECK: %gamma = spechls.gamma<"alias_check_x_distance_3">(%7, %c3_i32, %c2_i32) {}: i1, i32
    //CHECK: %8 = comb.icmp eq %1, %arg1 : i32
    //CHECK: %9 = comb.and %8, %4 : i1
    //CHECK: %gamma_0 = spechls.gamma<"alias_check_x_distance_2">(%9, %gamma, %c1_i32) {}: i1, i32
    //CHECK: %10 = comb.icmp eq %0, %arg1 : i32
    //CHECK: %11 = comb.and %10, %3 : i1
    //CHECK: %gamma_1 = spechls.gamma<"alias_check_x_distance_1">(%11, %gamma_0, %c0_i32) {}: i1, i32
    //CHECK: %12 = spechls.delay %mu by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %13 = spechls.delay %12 by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %14 = spechls.delay %13 by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %gamma_2 = spechls.gamma<"memory_speculation_x">(%gamma_1, %mu, %12, %13, %14) {}: i32, !spechls.array<i32, 16>
    //CHECK: %15 = spechls.alpha %mu[%arg2: i32], %arg3 if %arg4 : !spechls.array<i32, 16>
    //CHECK: %16 = spechls.load %gamma_2[%arg1 : i32] : <i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @twoRead
spechls.kernel @twoRead(%arr : !spechls.array<i32, 16>, %idxRead1 : i32, %idxRead2 : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 1} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    //CHECK: %0 = spechls.delay %arg3 by 1 if %true init %arg3 : i32 
    //CHECK: %1 = spechls.delay %arg5 by 1 if %true init %arg5 : i1 
    //CHECK: %2 = comb.icmp eq %0, %arg2 : i32
    //CHECK: %3 = comb.and %2, %1 : i1
    //CHECK: %4 = comb.icmp eq %0, %arg1 : i32
    //CHECK: %5 = comb.and %4, %1 : i1
    //CHECK: %6 = comb.or %5, %3 : i1
    //CHECK: %gamma = spechls.gamma<"alias_check_x_distance_1">(%6, %c1_i32, %c0_i32) {}: i1, i32
    //CHECK: %7 = spechls.delay %mu by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %gamma_0 = spechls.gamma<"memory_speculation_x">(%gamma, %mu, %7) {}: i32, !spechls.array<i32, 16>
    //CHECK: %8 = spechls.alpha %mu[%arg3: i32], %arg4 if %arg5 : !spechls.array<i32, 16>
    //CHECK: %9 = spechls.load %gamma_0[%arg1 : i32] : <i32, 16>
    //CHECK: %10 = spechls.load %gamma_0[%arg2 : i32] : <i32, 16>
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
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 1} : !spechls.array<i32, 16>
    %temp_arr = spechls.alpha %mu[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %temp_arr[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    //CHECK: %0 = spechls.delay %arg2 by 1 if %true init %arg2 : i32 
    //CHECK: %1 = spechls.delay %arg3 by 1 if %true init %arg3 : i32 
    //CHECK: %2 = spechls.delay %arg6 by 1 if %true init %arg6 : i1 
    //CHECK: %3 = spechls.delay %arg7 by 1 if %true init %arg7 : i1 
    //CHECK: %4 = comb.icmp eq %0, %arg1 : i32
    //CHECK: %5 = comb.and %4, %2 : i1
    //CHECK: %6 = comb.icmp eq %1, %arg1 : i32
    //CHECK: %7 = comb.and %6, %3 : i1
    //CHECK: %8 = comb.or %5, %7 : i1
    //CHECK: %gamma = spechls.gamma<"alias_check_x_distance_1">(%8, %c1_i32, %c0_i32) {}: i1, i32
    //CHECK: %9 = spechls.delay %mu by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %gamma_0 = spechls.gamma<"memory_speculation_x">(%gamma, %mu, %9) {}: i32, !spechls.array<i32, 16>
    //CHECK: %10 = spechls.alpha %mu[%arg2: i32], %arg4 if %arg6 : !spechls.array<i32, 16>
    //CHECK: %11 = spechls.alpha %10[%arg3: i32], %arg5 if %arg7 : !spechls.array<i32, 16>
    //CHECK: %12 = spechls.load %gamma_0[%arg1 : i32] : <i32, 16>
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
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 3} : !spechls.array<i32, 16>
    %temp_arr = spechls.alpha %mu[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %temp_arr[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    //CHECK: %0 = spechls.delay %arg3 by 1 if %true init %arg3 : i32 
    //CHECK: %1 = spechls.delay %arg4 by 1 if %true init %arg4 : i32 
    //CHECK: %2 = spechls.delay %0 by 1 if %true init %0 : i32 
    //CHECK: %3 = spechls.delay %1 by 1 if %true init %1 : i32 
    //CHECK: %4 = spechls.delay %2 by 1 if %true init %2 : i32 
    //CHECK: %5 = spechls.delay %3 by 1 if %true init %3 : i32 
    //CHECK: %6 = spechls.delay %arg7 by 1 if %true init %arg7 : i1 
    //CHECK: %7 = spechls.delay %arg8 by 1 if %true init %arg8 : i1 
    //CHECK: %8 = spechls.delay %6 by 1 if %true init %6 : i1 
    //CHECK: %9 = spechls.delay %7 by 1 if %true init %7 : i1 
    //CHECK: %10 = spechls.delay %8 by 1 if %true init %8 : i1 
    //CHECK: %11 = spechls.delay %9 by 1 if %true init %9 : i1 
    //CHECK: %12 = comb.icmp eq %4, %arg2 : i32
    //CHECK: %13 = comb.and %12, %10 : i1
    //CHECK: %14 = comb.icmp eq %5, %arg2 : i32
    //CHECK: %15 = comb.and %14, %11 : i1
    //CHECK: %16 = comb.or %13, %15 : i1
    //CHECK: %17 = comb.icmp eq %4, %arg1 : i32
    //CHECK: %18 = comb.and %17, %10 : i1
    //CHECK: %19 = comb.icmp eq %5, %arg1 : i32
    //CHECK: %20 = comb.and %19, %11 : i1
    //CHECK: %21 = comb.or %18, %20 : i1
    //CHECK: %22 = comb.or %21, %16 : i1
    //CHECK: %gamma = spechls.gamma<"alias_check_x_distance_3">(%22, %c3_i32, %c2_i32) {}: i1, i32
    //CHECK: %23 = comb.icmp eq %2, %arg2 : i32
    //CHECK: %24 = comb.and %23, %8 : i1
    //CHECK: %25 = comb.icmp eq %3, %arg2 : i32
    //CHECK: %26 = comb.and %25, %9 : i1
    //CHECK: %27 = comb.or %24, %26 : i1
    //CHECK: %28 = comb.icmp eq %2, %arg1 : i32
    //CHECK: %29 = comb.and %28, %8 : i1
    //CHECK: %30 = comb.icmp eq %3, %arg1 : i32
    //CHECK: %31 = comb.and %30, %9 : i1
    //CHECK: %32 = comb.or %29, %31 : i1
    //CHECK: %33 = comb.or %32, %27 : i1
    //CHECK: %gamma_0 = spechls.gamma<"alias_check_x_distance_2">(%33, %gamma, %c1_i32) {}: i1, i32
    //CHECK: %34 = comb.icmp eq %0, %arg2 : i32
    //CHECK: %35 = comb.and %34, %6 : i1
    //CHECK: %36 = comb.icmp eq %1, %arg2 : i32
    //CHECK: %37 = comb.and %36, %7 : i1
    //CHECK: %38 = comb.or %35, %37 : i1
    //CHECK: %39 = comb.icmp eq %0, %arg1 : i32
    //CHECK: %40 = comb.and %39, %6 : i1
    //CHECK: %41 = comb.icmp eq %1, %arg1 : i32
    //CHECK: %42 = comb.and %41, %7 : i1
    //CHECK: %43 = comb.or %40, %42 : i1
    //CHECK: %44 = comb.or %43, %38 : i1
    //CHECK: %gamma_1 = spechls.gamma<"alias_check_x_distance_1">(%44, %gamma_0, %c0_i32) {}: i1, i32
    //CHECK: %45 = spechls.delay %mu by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %46 = spechls.delay %45 by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %47 = spechls.delay %46 by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %gamma_2 = spechls.gamma<"memory_speculation_x">(%gamma_1, %mu, %45, %46, %47) {}: i32, !spechls.array<i32, 16>
    //CHECK: %48 = spechls.alpha %mu[%arg3: i32], %arg5 if %arg7 : !spechls.array<i32, 16>
    //CHECK: %49 = spechls.alpha %48[%arg4: i32], %arg6 if %arg8 : !spechls.array<i32, 16>
    //CHECK: %50 = spechls.load %gamma_2[%arg1 : i32] : <i32, 16>
    //CHECK: %51 = spechls.load %gamma_2[%arg2 : i32] : <i32, 16>
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
    %true = hw.constant 1 : i1
    %mu1 = spechls.mu<"x">(%arr1, %next_arr1) {dependenciesDistances = 3} : !spechls.array<i32, 16>
    %mu2 = spechls.mu<"y">(%arr2, %next_arr2) {dependenciesDistances = 2} : !spechls.array<i32, 16>
    %next_arr1 = spechls.alpha %mu1[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr2 = spechls.alpha %mu2[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    //CHECK: %0 = spechls.delay %arg4 by 1 if %true init %arg4 : i32 
    //CHECK: %1 = spechls.delay %0 by 1 if %true init %0 : i32 
    //CHECK: %2 = spechls.delay %1 by 1 if %true init %1 : i32 
    //CHECK: %3 = spechls.delay %arg8 by 1 if %true init %arg8 : i1 
    //CHECK: %4 = spechls.delay %3 by 1 if %true init %3 : i1 
    //CHECK: %5 = spechls.delay %4 by 1 if %true init %4 : i1 
    //CHECK: %6 = comb.icmp eq %2, %arg2 : i32
    //CHECK: %7 = comb.and %6, %5 : i1
    //CHECK: %gamma = spechls.gamma<"alias_check_x_distance_3">(%7, %c3_i32, %c2_i32) {}: i1, i32
    //CHECK: %8 = comb.icmp eq %1, %arg2 : i32
    //CHECK: %9 = comb.and %8, %4 : i1
    //CHECK: %gamma_0 = spechls.gamma<"alias_check_x_distance_2">(%9, %gamma, %c1_i32) {}: i1, i32
    //CHECK: %10 = comb.icmp eq %0, %arg2 : i32
    //CHECK: %11 = comb.and %10, %3 : i1
    //CHECK: %gamma_1 = spechls.gamma<"alias_check_x_distance_1">(%11, %gamma_0, %c0_i32) {}: i1, i32
    //CHECK: %12 = spechls.delay %mu by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %13 = spechls.delay %12 by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %14 = spechls.delay %13 by 1 if %true init %arg0 : !spechls.array<i32, 16> 
    //CHECK: %gamma_2 = spechls.gamma<"memory_speculation_x">(%gamma_1, %mu, %12, %13, %14) {}: i32, !spechls.array<i32, 16>
    %operand1 = spechls.load %mu1[%idxRead1 : i32] : !spechls.array<i32, 16>
    //CHECK: %15 = spechls.delay %arg5 by 1 if %true init %arg5 : i32 
    //CHECK: %16 = spechls.delay %15 by 1 if %true init %15 : i32 
    //CHECK: %17 = spechls.delay %arg9 by 1 if %true init %arg9 : i1 
    //CHECK: %18 = spechls.delay %17 by 1 if %true init %17 : i1 
    //CHECK: %19 = comb.icmp eq %16, %arg3 : i32
    //CHECK: %20 = comb.and %19, %18 : i1
    //CHECK: %gamma_3 = spechls.gamma<"alias_check_y_distance_2">(%20, %c2_i32, %c1_i32) {}: i1, i32
    //CHECK: %21 = comb.icmp eq %15, %arg3 : i32
    //CHECK: %22 = comb.and %21, %17 : i1
    //CHECK: %gamma_4 = spechls.gamma<"alias_check_y_distance_1">(%22, %gamma_3, %c0_i32) {}: i1, i32
    //CHECK: %23 = spechls.delay %mu_6 by 1 if %true init %arg1 : !spechls.array<i32, 16> 
    //CHECK: %24 = spechls.delay %23 by 1 if %true init %arg1 : !spechls.array<i32, 16> 
    //CHECK: %gamma_5 = spechls.gamma<"memory_speculation_y">(%gamma_4, %mu_6, %23, %24) {}: i32, !spechls.array<i32, 16>
    //CHECK: %25 = spechls.alpha %mu[%arg4: i32], %arg6 if %arg8 : !spechls.array<i32, 16>
    //CHECK: %26 = spechls.alpha %mu_6[%arg5: i32], %arg7 if %arg9 : !spechls.array<i32, 16>
    //CHECK: %27 = spechls.load %gamma_2[%arg2 : i32] : <i32, 16>
    //CHECK: %28 = spechls.load %gamma_5[%arg3 : i32] : <i32, 16>
    %operand2 = spechls.load %mu2[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}
