// RUN: spechls-opt --expose-control-flow-speculation="targetTask=task clockPeriod=10 targetsFile=%S/../../../codegen/caracAsic.json" %s | FileCheck %s

spechls.kernel @kernel(%x_init: i32, %y_init: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.task "task"(%x_in = %x_init, %y_in = %y_init): (i32, i32) -> !spechls.struct<"commit_type" {"enable": i1, "x": i32, "y": i32}> {
    %enabled = hw.constant 1 : i1
    %x = spechls.mu<"x">(%x_in, %x_gamma) : i32
    %y = spechls.mu<"y">(%y_in, %y_gamma) : i32
    %x_cond = spechls.call @x_cond(%x) {spechls.combDelay = 20.0: f64} : (i32) -> i1
    %y_cond = spechls.call @y_cond(%y) {spechls.combDelay = 20.0: f64} : (i32) -> i1
    %x_fast = spechls.call @x_fast(%x) : (i32) -> i32
    %y_fast = spechls.call @y_fast(%y) : (i32) -> i32
    %x_slow = spechls.call @x_slow(%x) {spechls.combDelay = 70.0: f64} : (i32) -> i32
    %y_slow = spechls.call @y_slow(%y) {spechls.combDelay = 70.0: f64} : (i32) -> i32
    %x_gamma = spechls.gamma<"x">(%x_cond, %x_fast, %x_slow) {spechls.profilingId = 0 : i64} : i1, i32
    %y_gamma = spechls.gamma<"y">(%y_cond, %y_fast, %y_slow) {spechls.profilingId = 1 : i64} : i1, i32
    spechls.commit %enabled, %x_gamma, %y_gamma : i1, i32, i32
  }
  %1 = spechls.field<"x"> %0 : !spechls.struct<"commit_type" {"enable": i1, "x": i32, "y": i32}>
  spechls.exit if %true with %1 : i32
}

// CHECK: spechls.fsm.state "Combined_0_Stall_0"
// CHECK: spechls.fsm.transition {{.*}} "new_mispec"
// CHECK: spechls.fsm.transition {{.*}} "canceled"
// CHECK: spechls.fsm.state "NewMispec_0"
