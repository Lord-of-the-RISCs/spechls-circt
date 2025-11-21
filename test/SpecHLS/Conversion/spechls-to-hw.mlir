// RUN: spechls-opt -split-input-file --spechls-to-hw --canonicalize --cse %s | spechls-opt | FileCheck %s

// CHECK-LABEL: hw.module @kernel
spechls.kernel @kernel() {
  %true = hw.constant 1 : i1
  // CHECK: hw.output
  spechls.exit if %true
}

//---

// CHECK-LABEL: @task
spechls.kernel @task(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %task1.enable, %task1.commit_val_0 = hw.instance "task1" @task1(arg1: %arg0: i32) -> (enable: i1, commit_val_0: i32)
  %0 = spechls.task "task1"(%arg1 = %in1) : (i32) -> !spechls.struct<"commit_type_0" {"enable": i1, "commit_val_0" : i32}> {
    %true = hw.constant 1 : i1
    spechls.commit %true, %arg1 : i1, i32
  }

  %1 = spechls.field<"commit_val_0"> %0 : !spechls.struct<"commit_type_0" {"enable": i1, "commit_val_0" : i32}>
  // CHECK: hw.output %task1.commit_val_0 : i32
  spechls.exit if %in2 with %1 : i32
}

//---

// CHECK-LABEL: @gamma
spechls.kernel @gamma(%cond: i1, %x: i32, %y: i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: comb.mux %arg0, %arg2, %arg1 : i32
  %0 = spechls.gamma<"x">(%cond, %x, %y) : i1, i32
  spechls.exit if %true with %0 : i32
}

//---

// CHECK-LABEL: @nary_gamma
spechls.kernel @nary_gamma(%cond: i2, %x: i32, %y: i32, %z: i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %[[array:.+]] = hw.array_create %arg3, %arg2, %arg1 : i32
  // CHECK: hw.array_get %[[array]][%arg0] : !hw.array<3xi32>, i2
  %0 = spechls.gamma<"x">(%cond, %x, %y, %z) : i2, i32
  spechls.exit if %true with %0 : i32
}

//---
