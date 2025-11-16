// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @slowfast
spechls.kernel @slowfast(%init : i32) -> i32
 attributes {
 spechls.max_scc=0,
 spechls.task_configs = [0 ,1 ,2 ,3],
 spechls.gamma =  [0 ,1 ,2 ,3]//,
 //spechls.config = #spechls.gamma_spec<"S0", 3, 0, 1, [0, 1, 2]>

 } {
        %false = hw.constant 1 : i1
        %mu = spechls.mu<"x">(%next, %init) : i32
        %slow = spechls.call @slow(%mu) : (i32) -> i32
        %fast = spechls.call @fast(%mu) : (i32) -> i32
        %cond = spechls.call @cond(%mu) : (i32) -> i32
        %exit = spechls.call @exit(%mu) : (i32) -> i1
        %next = spechls.gamma<"x">(%cond, %slow, %fast) : i32, i32
        spechls.exit if %exit with %next : i32
}
