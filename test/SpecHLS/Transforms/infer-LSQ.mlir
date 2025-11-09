//RUN: spechls-opt -split-input-file --infer-LSQ %s | spechls-opt | FileCheck %s 

// CHECK-LABEL: @simple
spechls.kernel@simple(%arr: !spechls.array<i32, 16>, %waddr : i32, %wval : i32, %we : i1,
    %raddr : i32, %rb : i32, %wc : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"arr">(%arr, %new_arr) : !spechls.array<i32, 16>
    %rb_mu = spechls.rollback<[5], 2>%mu,%rb,%wc : !spechls.array<i32, 16>, i32
    //CHECK: %1 = spechls.delay %arg3 by 1 if %true init %arg3 {} : i1
    //CHECK: %2 = spechls.delay %arg2 by 1 if %true init %arg2 {} : i32
    //CHECK: %3 = spechls.delay %arg1 by 1 if %true init %arg1 {} : i32
    //CHECK: %4 = spechls.rollback<[5], 2> %1, %arg5, %arg6 : i1, i32
    //CHECK: %5 = spechls.rollback<[5], 2> %2, %arg5, %arg6 : i32, i32
    //CHECK: %6 = spechls.rollback<[5], 2> %3, %arg5, %arg6 : i32, i32
    //CHECK: %7 = spechls.delay %4 by 1 if %true init %4 {} : i1
    //CHECK: %8 = spechls.delay %5 by 1 if %true init %5 {} : i32
    //CHECK: %9 = spechls.delay %6 by 1 if %true init %6 {} : i32
    //CHECK: %10 = spechls.alpha %0[%9: i32], %8 if %7 : !spechls.array<i32, 16>
    //CHECK: %11 = spechls.load %10[%arg4 : i32] : <i32, 16>
    %new_arr = spechls.alpha %rb_mu[%waddr : i32], %wval if %we : !spechls.array<i32, 16>
    %delay_arr = spechls.delay %rb_mu by 1 if %true init %arr : !spechls.array<i32, 16>
    %result = spechls.load %delay_arr[%raddr : i32] : <i32, 16>
  spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @doubleWrite
spechls.kernel@doubleWrite(%arr: !spechls.array<i32, 16>, %waddr1 : i32, %wval1 : i32, %we1 : i1,
  %waddr2 : i32, %wval2 : i32, %we2 : i1,
    %raddr : i32, %rb : i32, %wc : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"arr">(%arr, %new_arr) : !spechls.array<i32, 16>
    %rb_mu = spechls.rollback<[5], 2>%mu,%rb,%wc : !spechls.array<i32, 16>, i32
    //CHECK: %1 = spechls.delay %arg3 by 1 if %true init %arg3 {} : i1
    //CHECK: %2 = spechls.delay %arg2 by 1 if %true init %arg2 {} : i32
    //CHECK: %3 = spechls.delay %arg1 by 1 if %true init %arg1 {} : i32
    //CHECK: %4 = spechls.rollback<[5], 2> %1, %arg8, %arg9 : i1, i32
    //CHECK: %5 = spechls.rollback<[5], 2> %2, %arg8, %arg9 : i32, i32
    //CHECK: %6 = spechls.rollback<[5], 2> %3, %arg8, %arg9 : i32, i32
    //CHECK: %7 = spechls.delay %4 by 1 if %true init %4 {} : i1
    //CHECK: %8 = spechls.delay %5 by 1 if %true init %5 {} : i32
    //CHECK: %9 = spechls.delay %6 by 1 if %true init %6 {} : i32
    //CHECK: %10 = spechls.alpha %0[%9: i32], %8 if %7 : !spechls.array<i32, 16>
    //CHECK: %11 = spechls.delay %arg6 by 1 if %true init %arg6 {} : i1
    //CHECK: %12 = spechls.delay %arg5 by 1 if %true init %arg5 {} : i32
    //CHECK: %13 = spechls.delay %arg4 by 1 if %true init %arg4 {} : i32
    //CHECK: %14 = spechls.rollback<[5], 2> %11, %arg8, %arg9 : i1, i32
    //CHECK: %15 = spechls.rollback<[5], 2> %12, %arg8, %arg9 : i32, i32
    //CHECK: %16 = spechls.rollback<[5], 2> %13, %arg8, %arg9 : i32, i32
    //CHECK: %17 = spechls.delay %14 by 1 if %true init %14 {} : i1
    //CHECK: %18 = spechls.delay %15 by 1 if %true init %15 {} : i32
    //CHECK: %19 = spechls.delay %16 by 1 if %true init %16 {} : i32
    //CHECK: %20 = spechls.alpha %10[%19: i32], %18 if %17 : !spechls.array<i32, 16>
    //CHECK: %21 = spechls.load %20[%arg7 : i32] : <i32, 16>
    %tmp_arr = spechls.alpha %rb_mu[%waddr1 : i32], %wval1 if %we1 : !spechls.array<i32, 16>
    %new_arr = spechls.alpha %tmp_arr[%waddr2 : i32], %wval2 if %we2 : !spechls.array<i32, 16>
    %delay_arr = spechls.delay %rb_mu by 1 if %true init %arr : !spechls.array<i32, 16>
    %result = spechls.load %delay_arr[%raddr : i32] : <i32, 16>
  spechls.exit if %true with %result : i32
}

// CHECK-LABEL: @doubleRead
spechls.kernel@doubleRead(%arr: !spechls.array<i32, 16>, %waddr : i32, %wval : i32, %we : i1,
    %raddr1 : i32, %raddr2 : i32, %rb : i32, %wc : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"arr">(%arr, %new_arr) : !spechls.array<i32, 16>
    %rb_mu = spechls.rollback<[5], 2>%mu,%rb,%wc : !spechls.array<i32, 16>, i32
    //CHECK: %1 = spechls.delay %arg3 by 1 if %true init %arg3 {} : i1
    //CHECK: %2 = spechls.delay %arg2 by 1 if %true init %arg2 {} : i32
    //CHECK: %3 = spechls.delay %arg1 by 1 if %true init %arg1 {} : i32
    //CHECK: %4 = spechls.rollback<[5], 2> %1, %arg6, %arg7 : i1, i32
    //CHECK: %5 = spechls.rollback<[5], 2> %2, %arg6, %arg7 : i32, i32
    //CHECK: %6 = spechls.rollback<[5], 2> %3, %arg6, %arg7 : i32, i32
    //CHECK: %7 = spechls.delay %4 by 1 if %true init %4 {} : i1
    //CHECK: %8 = spechls.delay %5 by 1 if %true init %5 {} : i32
    //CHECK: %9 = spechls.delay %6 by 1 if %true init %6 {} : i32
    //CHECK: %10 = spechls.alpha %0[%9: i32], %8 if %7 : !spechls.array<i32, 16>
    //CHECK: %11 = spechls.load %10[%arg4 : i32] : <i32, 16>
    //CHECK: %12 = spechls.load %10[%arg5 : i32] : <i32, 16>
    %new_arr = spechls.alpha %rb_mu[%waddr : i32], %wval if %we : !spechls.array<i32, 16>
    %delay_arr = spechls.delay %rb_mu by 1 if %true init %arr : !spechls.array<i32, 16>
    %op1 = spechls.load %delay_arr[%raddr1 : i32] : <i32, 16>
    %op2 = spechls.load %delay_arr[%raddr2 : i32] : <i32, 16>
    %result = comb.add %op1, %op2 : i32
  spechls.exit if %true with %result : i32
}
