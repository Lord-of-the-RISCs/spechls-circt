//RUN: spechls-opt --insert-instr="instrs=0xff" %s | spechls-opt | FileCheck %s

module {
  //CHECK-LABEL @analyzer
  //CHECK: wcet.core_instance @test_core(%c255_i32, %3#0, %4, %3#1)
  wcet.core @analyzer() -> () attributes {"wcet.analysis"} { %init0 = wcet.init @delay0 : i32
      %init1 = wcet.init @delay1 : i32
      %init2 = wcet.init @delay2 : i32
      %dummy:3 = wcet.dummy %init0, %init1, %init2 : i32, i32, i32 -> i32, i32, i32 {wcet.next, wcet.penalties = 1 : i32}
    wcet.commit
  }

  wcet.core @test_core(%instr : i32 {wcet.instrNb = 0 : i32}, %out0 : i32 {wcet.nbPred = 0 : i32}, %out1 : i32 {wcet.nbPred = 1 : i32}, %out2 : i32 {wcet.nbPred = 2 : i32}) -> (i32, i32, i32) attributes {wcet.cpuCore}{
    %op1 = comb.extract %instr from 0 : (i32) -> i16
      %op2 = comb.extract %instr from 16 : (i32) -> i16
      %result = comb.concat %op1, %op2 : i16, i16
      wcet.commit %result, %out0, %out1 : i32, i32, i32
  }
}
