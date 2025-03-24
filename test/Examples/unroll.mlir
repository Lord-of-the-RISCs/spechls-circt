module {

   hw.module.extern @fetch(in %pc: i32, out "instr" : i32) 

    hw.module @pipeline() attributes {"#pragma" = "UNROLL_NODE"}
    {
      // Constant and utils declaration
        %t = hw.constant 0x1 : i1
        %2 = hw.constant 0x4 : i32
        
      // Instruction fetch logic with pc update
        %0 = SpecHLS.init @pc_0 : i32
        %3 = SpecHLS.mu @mu_pc : %0, %4 : i32
        %4 = comb.add %2, %3 : i32
        %5 = hw.instance "instr" @fetch(pc: %3: i32) -> (instr: i32) {"#pragma" = "entry_point"}
        
      // Instruction decode logic
        %6 = comb.extract %5 from 0 : (i32) -> i1
        %7 = comb.extract %5 from 1 : (i32) -> i31
        %13 = comb.concat %7, %t : i31, i1
      
      // Combination logique
        %1 = SpecHLS.init @x_0 : i32
        %8 = SpecHLS.mu @mu_x : %1, %12 : i32
        %9 = comb.mul %8, %13 : i32
        %10 = comb.add %8, %13 : i32
        %11 = SpecHLS.delay %t -> %9 by 2:i32 
        %12 = SpecHLS.gamma %6 ? %10,%11 :i32
    }

}
