// RUN: spechls-opt --infer-speculation-fsm %s | FileCheck %s

spechls.kernel @kernel(%x_init: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.task "task"(%in = %x_init) : (i32) -> !spechls.struct<"commit_type" {"enable": i1, "value": i32}> attributes {spechls.speculation_config = [{gamma_id = "x", cond_latency = 2 : i64, fast_selector = 0 : i64, poison_speculation_ids = [], slow_paths = [{selector = 1 : i64, latency = 7 : i64, rewind = 0 : i64, rbwe = 1 : i64, rollback_mu_ids = [], rollback_array_ids = [], rollback_gamma_ids = [], rollback_depth = 7 : i64}]}]} {
    %enabled = hw.constant 1 : i1
    %cond = hw.constant 0 : i1
    %fast = hw.constant 0 : i32
    %slow = hw.constant 1 : i32
    %gamma = spechls.gamma<"x">(%cond, %fast, %slow) : i1, i32
    spechls.commit %enabled, %gamma : i1, i32
  }
  %1 = spechls.field<"value"> %0 : !spechls.struct<"commit_type" {"enable": i1, "value": i32}>
  spechls.exit if %true with %1 : i32
}

// CHECK: spechls.fsm.machine @fsm_x0 : (i32, i1) -> i32
// CHECK: spechls.fsm.state "Init_0"
// CHECK: spechls.fsm.state "Init_1"
// CHECK: spechls.fsm.state "Proceed"
// CHECK: spechls.fsm.transition "0_0_Rollback" "normal" guard {
// CHECK-NEXT: spechls.fsm.input 1 : i1
// CHECK: arith.cmpi eq
// CHECK: spechls.fsm.state "0_0_Rollback"
