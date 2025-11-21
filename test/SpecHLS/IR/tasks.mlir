// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @toplevel
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: i32, %[[arg2:[a-zA-Z0-9]+]]: i1
spechls.kernel @toplevel(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %[[task:.+]] = spechls.task "task1"(%[[targ:.+]] = %[[arg1]]) : (i32) -> !spechls.struct<"commit_type_0" { "enable" : i1, "commit_val_0" : i32 }>
  %0 = spechls.task "task1"(%arg1 = %in1) : (i32) -> !spechls.struct<"commit_type_0" { "enable" : i1, "commit_val_0" : i32 }> {
    %true = hw.constant 1 : i1
    // CHECK: spechls.commit %true, %[[targ]] : i1, i32
    spechls.commit %true, %arg1 : i1, i32
  }
  // CHECK: %[[res:.+]] = spechls.field<"commit_val_0"> %[[task]] : <"commit_type_0" { "enable" : i1, "commit_val_0" : i32 }>
  %1 = spechls.field<"commit_val_0"> %0 : !spechls.struct<"commit_type_0" { "enable" : i1, "commit_val_0" : i32 }>
  // CHECK: spechls.exit if %[[arg2]] with %[[res]] : i32
  spechls.exit if %in2 with %1 : i32
}
