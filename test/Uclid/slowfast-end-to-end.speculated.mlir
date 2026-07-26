module {
  spechls.fsm.machine @fsm_x0 : (i32, i1) -> (i32, i1, i1, i32, i32, i32, i1, i32, i32, i32, i32) {
    spechls.fsm.state "Init_0" output {
      spechls.fsm.output ["nextInput", "commit", "muRollBack", "arrayRollBack", "rewind", "rbwe", "gammaRollBack_x0", "selSlowPath_x0", "stall_x0_0", "stall_x0_1"] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    } transitions {
      spechls.fsm.transition "Init_1" "normal"
    }
    spechls.fsm.state "Init_1" output {
      spechls.fsm.output ["nextInput", "commit", "muRollBack", "arrayRollBack", "rewind", "rbwe", "gammaRollBack_x0", "selSlowPath_x0", "stall_x0_0", "stall_x0_1"] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    } transitions {
      spechls.fsm.transition "Proceed" "normal"
    }
    spechls.fsm.state "Proceed" output {
      spechls.fsm.output ["nextInput", "commit", "muRollBack", "arrayRollBack", "rewind", "rbwe", "gammaRollBack_x0", "selSlowPath_x0", "stall_x0_0", "stall_x0_1"] [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    } transitions {
      spechls.fsm.transition "Proceed" "normal"
      spechls.fsm.transition "0_0_Rollback" "normal" guard {
        %0 = spechls.fsm.input 1 : i1
        %true = arith.constant true
        %1 = arith.cmpi eq, %0, %true : i1
        spechls.fsm.return %1
      }
    }
    spechls.fsm.state "0_0_Rollback" output {
      spechls.fsm.output ["nextInput", "commit", "muRollBack", "arrayRollBack", "rewind", "rbwe", "gammaRollBack_x0", "selSlowPath_x0", "stall_x0_0", "stall_x0_1"] [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    } transitions {
      spechls.fsm.transition "0_0_Fill_0" "normal"
    }
    spechls.fsm.state "0_0_Fill_0" output {
      spechls.fsm.output ["nextInput", "commit", "muRollBack", "arrayRollBack", "rewind", "rbwe", "gammaRollBack_x0", "selSlowPath_x0", "stall_x0_0", "stall_x0_1"] [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    } transitions {
      spechls.fsm.transition "Proceed" "normal"
    }
  }
  spechls.kernel @slowfast(%arg0: i32) -> i32 {
    %true = hw.constant true
    %0 = spechls.task "slowfast"(%arg1 = %arg0) : (i32) -> !spechls.struct<"commit_type" { "enable" : i1, "value" : i32 }>  attributes {spechls.combDelay = 0.000000e+00 : f64, spechls.speculation_config = [{cond_latency = 2 : i64, fast_selector = 0 : i64, gamma_id = "x", poison_speculation_ids = [], slow_paths = [{latency = 3 : i64, rbwe = 1 : i64, rewind = 0 : i64, rollback_array_ids = [], rollback_depth = 3 : i64, rollback_gamma_ids = [], rollback_mu_ids = [], selector = 1 : i64}]}], spechls.speculativeTask}{
      %c0_i32 = hw.constant 0 : i32
      %mu = spechls.mu<"fsm_x0State">(%c0_i32, %2#0) {}: i32
      spechls.fsm.instance "fsm_x0" @fsm_x0
      %2:11 = spechls.fsm.trigger "fsm_x0"(%mu, %10) : (i32, i1) -> (i32, i1, i1, i32, i32, i32, i1, i32, i32, i32, i32)
      %3 = spechls.rollback<[2], 0> %gamma, %2#7, %2#6 : i32, i32
      %4 = spechls.rollback<[2], 0> %mu_6, %2#3, %2#6 : i32, i32
      %true_0 = hw.constant true
      %5 = spechls.delay %3 by 2 if %true_0 {} : i32
      %c0_i32_1 = hw.constant 0 : i32
      %6 = comb.icmp eq %2#10, %c0_i32_1 : i32
      %c0_i32_2 = hw.constant 0 : i32
      %7 = comb.icmp eq %2#10, %c0_i32_2 : i32
      %true_3 = hw.constant true
      %8 = spechls.delay %slow by 1 if %6 {} : i32
      %9 = spechls.delay %12 by 1 if %7 {} : i32
      %true_4 = hw.constant true
      %10 = spechls.delay %11 by 1 if %true_4 {} : i1
      %true_5 = hw.constant true {spechls.combDelay = 0.000000e+00 : f64}
      %mu_6 = spechls.mu<"x">(%arg1, %3) {spechls.combDelay = 0.000000e+00 : f64}: i32
      %cond = spechls.call @cond(%4) {spechls.combDelay = 1.000000e+01 : f64} : (i32) -> i1
      %11 = spechls.sync %cond {spechls.combDelay = 1.000000e+01 : f64} : i1
      %fast = spechls.call @fast(%4) {spechls.combDelay = 0.000000e+00 : f64} : (i32) -> i32
      %slow = spechls.call @slow(%4) {spechls.combDelay = 1.000000e+01 : f64} : (i32) -> i32
      %12 = spechls.sync %8 {spechls.combDelay = 1.000000e+01 : f64} : i32
      %13 = spechls.sync %9 {spechls.combDelay = 1.000000e+01 : f64} : i32
      %gamma = spechls.gamma<"x">(%2#8, %fast, %13) {spechls.combDelay = 1.096989 : f64, spechls.probability = 9.000000e-01 : f64, spechls.resolveDelay = 2 : i32, spechls.speculation = 1 : i32}: i32, i32
      spechls.commit %2#2, %5 : i1, i32 {spechls.combDelay = 0.000000e+00 : f64}
    }
    %1 = spechls.field<"value"> %0 : <"commit_type" { "enable" : i1, "value" : i32 }>
    spechls.exit if %true with %1 : i32
  }
}
