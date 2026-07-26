// RUN: mkdir -p %S/output
// RUN: spechls-opt --expose-control-flow-speculation="targetTask=slowfast clockPeriod=10 targetsFile=%S/../../../codegen/caracAsic.json use-existing-speculation=1" %s > %S/output/slowfast-end-to-end.speculated.mlir
// RUN: FileCheck %s < %S/output/slowfast-end-to-end.speculated.mlir
// RUN: spechls-translate --spechls-to-cpp %S/output/slowfast-end-to-end.speculated.mlir > %S/output/slowfast-end-to-end.cpp
// RUN: FileCheck %s --check-prefix=CPP < %S/output/slowfast-end-to-end.cpp

// The probability is input metadata. The explicit selector below is the fixed
// configuration: speculation = 1 selects gamma input zero, the fast path.
spechls.kernel @slowfast(%x_init: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.task "slowfast"(%in = %x_init) : (i32) -> !spechls.struct<"commit_type" {"enable": i1, "value": i32}> {
    %enabled = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%in, %gamma) : i32
    // Two clock cycles at a 10-unit target clock.
    %cond = spechls.call @cond(%mu) {spechls.combDelay = 20.0 : f64} : (i32) -> i1
    // No combDelay means the fast result is available in the current cycle.
    %fast = spechls.call @fast(%mu) : (i32) -> i32
    // Three clock cycles at the same target clock.
    %slow = spechls.call @slow(%mu) {spechls.combDelay = 30.0 : f64} : (i32) -> i32
    %gamma = spechls.gamma<"x">(%cond, %fast, %slow) {spechls.probability = 0.900000 : f64, spechls.speculation = 1 : i32} : i1, i32
    spechls.commit %enabled, %gamma : i1, i32
  }
  %1 = spechls.field<"value"> %0 : !spechls.struct<"commit_type" {"enable": i1, "value": i32}>
  spechls.exit if %true with %1 : i32
}

// CHECK: spechls.fsm.machine @fsm_x0 : (i32, i1) ->
// CHECK: spechls.fsm.state "Init_0"
// CHECK: spechls.fsm.state "Init_1"
// CHECK: spechls.fsm.state "Proceed" output {
// CHECK-NEXT: spechls.fsm.output ["nextInput", "commit"
// CHECK: spechls.fsm.transition "0_0_Rollback" "normal" guard {
// CHECK-NEXT: spechls.fsm.input 1 : i1
// CHECK: arith.cmpi eq
// CHECK: spechls.fsm.state "0_0_Rollback"
// The shared config normalizes the selected fast input to zero-based selector 0.
// CHECK: spechls.speculation_config = [{cond_latency = 2 : i64, fast_selector = 0 : i64, gamma_id = "x"
// CHECK-SAME: slow_paths = [{latency = 3 : i64
// CHECK: spechls.fsm.instance "fsm_x0" @fsm_x0
// CHECK: spechls.fsm.trigger "fsm_x0"(%mu, %{{.*}}) : (i32, i1) ->

// CPP: static void fsm_fsm_x0_trigger
// CPP: output6 = 1;
// CPP: input1 == 1
// CPP: void slowfast(unsigned int &);
