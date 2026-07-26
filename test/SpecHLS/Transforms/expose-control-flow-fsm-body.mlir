// RUN: spechls-opt --expose-control-flow-speculation="targetTask=task clockPeriod=10 targetsFile=%S/../../../codegen/caracAsic.json use-existing-speculation=1" %s | FileCheck %s
// RUN: spechls-opt --expose-control-flow-speculation="targetTask=task clockPeriod=10 targetsFile=%S/../../../codegen/caracAsic.json use-existing-speculation=1" %s | spechls-translate --spechls-to-cpp | FileCheck %s --check-prefix=CPP

spechls.kernel @kernel(%x_init: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.task "task"(%in = %x_init): (i32) -> !spechls.struct<"commit_type" {"enable": i1, "value": i32}> {
    %enabled = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%in, %gamma) : i32
    %cond = spechls.call @cond(%mu) {spechls.combDelay = 20.0: f64} : (i32) -> i1
    %fast = spechls.call @fast(%mu) : (i32) -> i32
    %slow = spechls.call @slow(%mu) {spechls.combDelay = 70.0: f64} : (i32) -> i32
    %gamma = spechls.gamma<"x">(%cond, %fast, %slow) {spechls.speculation = 1 : i32} : i1, i32
    spechls.commit %enabled, %gamma : i1, i32
  }
  %1 = spechls.field<"value"> %0 : !spechls.struct<"commit_type" {"enable": i1, "value": i32}>
  spechls.exit if %true with %1 : i32
}

// CHECK: spechls.fsm.state "Init_0"
// CHECK: spechls.fsm.state "Proceed"
// CHECK: spechls.fsm.transition "0_0_Stall_0" "normal" guard {
// CHECK-NEXT: spechls.fsm.input 1 : i1
// CHECK: arith.cmpi eq
// CHECK: spechls.fsm.state "0_0_Stall_0"
// CHECK: spechls.fsm.state "0_0_Rollback"
// CHECK: spechls.speculation_config = [{cond_latency = 2 : i64, fast_selector = 0 : i64, gamma_id = "x"

// CPP: static void fsm_fsm_x0_trigger
// CPP: output6 = 1;
// CPP: input1 == 1
