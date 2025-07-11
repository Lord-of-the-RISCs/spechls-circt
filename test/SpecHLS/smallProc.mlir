module {
  spechls.kernel @kernel(%arg0 : i8, %arg1 : !spechls.array<i8, 256>, %arg2 : !spechls.array<i32, 4>) -> i8 {
        %true = hw.constant 1 : i1
        %task = spechls.task "core" (%pc = %arg0, %mem = %arg1, %x = %arg2, %tr = %true) :
          (i8, !spechls.array<i8, 256>, !spechls.array<i32, 4>, i1) -> i8 {
            %c1 = hw.constant 1 : i8
            %pc0 = spechls.mu <"pc">(%pc, %pcNext) : i8
            %x0 = spechls.mu <"x">(%x, %xNext) : !spechls.array<i32, 4>
            %instr = spechls.load %mem[%pc0 : i8] {"#pragma" = "WCET fetch"} : !spechls.array<i8, 256> 
            %pcNext = comb.add %pc0, %c1 : i8
            %op = comb.extract %instr from 0 : (i8) -> i1
            %rs1 = comb.extract %instr from 1 : (i8) -> i2
            %rs2 = comb.extract %instr from 3 : (i8) -> i2
            %rd = comb.extract %instr from 5 : (i8) -> i2
            %rd0 = spechls.delay %rd by 1 : i2
            %operand1 = spechls.load %x0[%rs1 : i2] : !spechls.array<i32, 4>
            %operand2 = spechls.load %x0[%rs2 : i2] : !spechls.array<i32, 4>
            %addResult = comb.add %operand1, %operand2 : i32
            %mulResult = comb.mul %operand1, %operand2 : i32
            %mulResultDelayed = wcet.penalty %mulResult by 2 : i32
            %result = spechls.gamma<"result">(%op, %addResult, %mulResultDelayed) : i1, i32
            %xNext = spechls.alpha %x0[%rd : i2], %result if %tr : !spechls.array<i32, 4> 
            spechls.commit %tr, %pc0 : i8
          }
        spechls.exit if %true with %task : i8
      }

  }
