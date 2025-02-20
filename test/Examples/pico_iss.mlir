module {
hw.module.extern @alu( in %fun : i8,  in %a : i32,  in %b : i32 , out "out0" : i32 )

hw.module.extern @mult( in %a : i32,  in %b : i32 , out "out0" : i32 )

hw.module.extern @target( in %ir : i32,  in %base : i32 , out "out0" : i32 )

hw.module.extern @encoder_4( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1,  in %in_3 : i1 , out "out0" : ui2 )

hw.module.extern @bv2toint( in %in_0 : i1,  in %in_1 : i1 , out "out0" : ui2 )

hw.module.extern @bv3toint( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1 , out "out0" : ui3 )

hw.module @SpecSCC_15(out "out_0": i1) attributes {"#pragma" = "UNROLL_NODE"} {
	%t1 = SpecHLS.init @regs : memref<32xi32> 
	%t2 = SpecHLS.mu @regs : %t1,%t3 : memref<32xi32> 
	%t4 = SpecHLS.init @pc : i32 
	%t5 = SpecHLS.mu @pc : %t4,%t6 : i32 
	%t7 = SpecHLS.init @isHalted : i1 
	%t8 = SpecHLS.mu @isHalted : %t7,%t9 : i1 
	%t10 = SpecHLS.init @mem : memref<256xi32> 
	%t11 = SpecHLS.mu @mem : %t10,%t12 : memref<256xi32> 
	%t13_idx = arith.index_cast %t5 : i32 to index
	%t13 = SpecHLS.read %t11:memref<256xi32>  [%t13_idx] {"#pragma" = "entry_point"} 
	%t14 = hw.constant 127 : i32 
	%t15 = comb.and %t13,%t14 : i32 
	%t16 = hw.constant 15 : i32 
	%t17 = comb.shrs %t13,%t16 : i32 
	%t18 = hw.constant 31 : i32 
	%t19 = comb.and %t17,%t18 : i32 
	%t20_idx = arith.index_cast %t19 : i32 to index
	%t20 = SpecHLS.read %t2:memref<32xi32>  [%t20_idx]   
	%t21 = hw.constant 20 : i32 
	%t22 = comb.shrs %t13,%t21 : i32 
	%t23 = comb.and %t22,%t18 : i32 
	%t24_idx = arith.index_cast %t23 : i32 to index
	%t24 = SpecHLS.read %t2:memref<32xi32>  [%t24_idx]   
	%t25 = hw.constant 7 : i32 
	%t26 = comb.shrs %t13,%t25 : i32 
	%t27 = comb.and %t26,%t18 : i32 
	%t28 = hw.constant 4095 : i32 
	%t29 = comb.and %t22,%t28 : i32 
	%t30 = hw.constant 4064 : i32 
	%t31 = comb.and %t22,%t30 : i32 
	%t32 = comb.add %t27,%t31 : i32 
	%t33 = hw.constant 12 : i32 
	%t34 = comb.shrs %t13,%t33 : i32 
	%t35 = comb.and %t34,%t25 : i32 
	%t36 = hw.constant 1 : i32 
	%t37 = comb.add %t5,%t36 : i32 
	%t35_c = SpecHLS.cast %t35:i32  to i8
	
	%t20_c = SpecHLS.cast %t20:i32  to i32
	
	%t29_c = SpecHLS.cast %t29:i32  to i32
					%t38 = hw.instance "%33" @alu  ( fun :  %t35_c : i8,  a :  %t20_c : i32,  b :  %t29_c : i32 ) -> ( out0 : i32 ) 
	%t39 = hw.constant 19 : i32 
	%t15_c = builtin.unrealized_conversion_cast %t15 : i32 to i32					
	%t39_c = builtin.unrealized_conversion_cast %t39 : i32 to i32					
	%t40 = comb.icmp eq %t15_c, %t39_c : i32
	%t41 = hw.constant 25 : i32 
	%t42 = comb.shrs %t13,%t41 : i32 
	%t43 = hw.constant 0 : i32 
	%t42_c = builtin.unrealized_conversion_cast %t42 : i32 to i32					
	%t43_c = builtin.unrealized_conversion_cast %t43 : i32 to i32					
	%t44 = comb.icmp eq %t42_c, %t43_c : i32
	
	
	%t24_c = SpecHLS.cast %t24:i32  to i32
					%t45 = hw.instance "%40" @alu  ( fun :  %t35_c : i8,  a :  %t20_c : i32,  b :  %t24_c : i32 ) -> ( out0 : i32 ) 
	
					%t46 = hw.instance "%41" @mult  ( a :  %t20_c : i32,  b :  %t24_c : i32 ) -> ( out0 : i32 ) 
	%t47 = hw.constant 51 : i32 
	%t47_c = builtin.unrealized_conversion_cast %t47 : i32 to i32					
	%t48 = comb.icmp eq %t15_c, %t47_c : i32
	%t49 = comb.add %t20,%t29 : i32 
	%t50_idx = arith.index_cast %t49 : i32 to index
	%t50 = SpecHLS.read %t11:memref<256xi32>  [%t50_idx]   
	%t51 = hw.constant 3 : i32 
	%t51_c = builtin.unrealized_conversion_cast %t51 : i32 to i32					
	%t52 = comb.icmp eq %t15_c, %t51_c : i32
	%t53 = comb.add %t20,%t32 : i32 
	%t54 = hw.constant 35 : i32 
	%t54_c = builtin.unrealized_conversion_cast %t54 : i32 to i32					
	%t55 = comb.icmp eq %t15_c, %t54_c : i32
	%t13_c = SpecHLS.cast %t13:i32  to i32
	
	%t5_c = SpecHLS.cast %t5:i32  to i32
					%t56 = hw.instance "%51" @target  ( ir :  %t13_c : i32,  base :  %t5_c : i32 ) -> ( out0 : i32 ) 
	%t57 = hw.constant 99 : i32 
	%t57_c = builtin.unrealized_conversion_cast %t57 : i32 to i32					
	%t58 = comb.icmp eq %t15_c, %t57_c : i32
	%t59 = hw.constant 1 : i1 
	%t60 = arith.ori  %t40,%t48 : i1 
	%t9 = SpecHLS.gamma @isHalted %t60 ? %t59,%t8 :i1 
	%t61 = hw.constant 0 : i1 
	%t62 = SpecHLS.init @guard : i1 
	%t63 = SpecHLS.mu @guard : %t62,%t64 : i1 
	%t65 = arith.andi %t55,%t63 : i1 
	%t66 = comb.icmp slt %t20_c, %t24_c : i32
	%t67_0 = hw.constant 0 : i1
	%t67 = comb.icmp eq %t67_0,%t66: i1
	%t68_0 = hw.constant 0 : i1
	%t68 = comb.icmp eq %t68_0,%t67: i1
	%t69_0 = hw.constant 0 : i1
	%t69 = comb.icmp eq %t69_0,%t44: i1
	%t70_0 = hw.constant 0 : i1
	%t70 = comb.icmp eq %t70_0,%t60: i1
	%t71 = arith.andi %t48,%t44 : i1 
	%t72 = arith.andi %t58,%t67 : i1 
	%t73 = arith.andi %t58,%t68 : i1 
	%t74 = arith.andi %t48,%t69 : i1 
	%t75 = arith.ori  %t71,%t74 : i1 
	%t76 = arith.ori  %t72,%t73 : i1 
	%t77 = arith.andi %t40,%t63 : i1 
	%t78 = arith.andi %t52,%t63 : i1 
	%t79 = arith.ori  %t55,%t76 : i1 
	%t80 = arith.ori  %t79,%t70 : i1 
  %t81_c0 = hw.constant 0 : i2
  %t81_c1 = hw.constant 1 : i2
  %t81_c2 = hw.constant 2 : i2
  %t81_c3 = hw.constant 3 : i2
  %t81_m0 = comb.mux %t75, %t81_c2, %t81_c3 : i2
  %t81_m1 = comb.mux %t40, %t81_c1, %t81_m0 : i2
  %t81 = comb.mux %t80, %t81_c0, %t81_m1 : i2
	%t82_0 = hw.constant 0 : i1
	%t82 = comb.icmp eq %t82_0,%t72: i1
	%t83 = arith.andi %t76,%t72 : i1 
	%t84 = arith.andi %t76,%t82 : i1 
	
  %t85 = comb.concat %t84, %t83 : i1, i1

	%t86 = comb.extract %t81 from 0 : (i2)->i1 
	%t87 = comb.extract %t81 from 1 : (i2)->i1 
	%t88_0 = hw.constant 0 : i1
	%t88 = comb.icmp eq %t88_0,%t74: i1
	%t89_0 = hw.constant 0 : i1
	%t89 = comb.icmp eq %t89_0,%t86: i1
	%t90 = arith.andi %t74,%t89 : i1 
	%t91 = arith.andi %t87,%t90 : i1 
	%t92 = arith.andi %t87,%t86 : i1 
	%t93_0 = hw.constant 0 : i1
	%t93 = comb.icmp eq %t93_0,%t92: i1
	%t94 = arith.andi %t88,%t87 : i1 
	%t95 = arith.ori  %t86,%t94 : i1 
	%t96 = arith.andi %t93,%t95 : i1 
	%t97 = arith.ori  %t92,%t94 : i1 
	
  %t98_t = comb.concat %t96, %t97 : i1, i1
  %t98 = comb.concat %t98_t, %t91 : i2, i1
	%t99 = SpecHLS.gamma @merge__1 %t100 ? %t43,%t27,%t27,%t27,%t27 :i32 
	%t101_0 = hw.constant 0 : i1
	%t101 = comb.icmp eq %t101_0,%t9: i1
	%t102 = arith.andi %t63,%t101 : i1 
	%t64 = SpecHLS.gamma @guard %t102 ? %t61,%t59 :i1 
	%t103 = arith.andi %t71,%t63 : i1 
	%t104 = arith.andi %t74,%t63 : i1 
	%t105 = SpecHLS.gamma @merge__0 %t106 ? %t61,%t77,%t78,%t103,%t104 :i1 
	%t107 = SpecHLS.gamma @merge__2 %t108 ? %t43,%t109,%t110,%t111,%t112 :i32 
	%t3_idx = arith.index_cast %t99 : i32 to index
	%t3 = SpecHLS.alpha @regs : %t105 -> %t2 [%t3_idx], %t107: memref<32xi32>  
	%t6 = SpecHLS.gamma @nextpc %t113 ? %t37,%t37,%t114 :i32 
	%t115 = SpecHLS.gamma @writeValue_mem %t116 ? %t43,%t117 :i32 
	%t118 = SpecHLS.gamma @writeAdress_mem %t119 ? %t43,%t120 :i32 
	%t121 = SpecHLS.gamma @writeEnable_mem %t122 ? %t61,%t123 :i1 
	%t12_idx = arith.index_cast %t118 : i32 to index
	%t12 = SpecHLS.alpha @mem : %t121 -> %t11 [%t12_idx], %t115: memref<256xi32>  
	%t124 = hw.constant 1 : i1 
	%t100 = SpecHLS.delay %t124 -> %t98 by 1:i3 
	%t125 = hw.constant 1 : i1 
	%t106 = SpecHLS.delay %t125 -> %t98 by 1:i3 
	%t126 = hw.constant 1 : i1 
	%t109 = SpecHLS.delay %t126 -> %t38 by 2:i32 
	%t127 = hw.constant 1 : i1 
	%t110 = SpecHLS.delay %t127 -> %t50 by 6:i32 
	%t128 = hw.constant 1 : i1 
	%t111 = SpecHLS.delay %t128 -> %t45 by 2:i32 
	%t129 = hw.constant 1 : i1 
	%t112 = SpecHLS.delay %t129 -> %t46 by 6:i32 
	%t130 = hw.constant 1 : i1 
	%t108 = SpecHLS.delay %t130 -> %t98 by 1:i3 
	%t131 = hw.constant 1 : i1 
	%t114 = SpecHLS.delay %t131 -> %t56 by 6:i32 
	%t132 = hw.constant 1 : i1 
	%t113 = SpecHLS.delay %t132 -> %t85 by 5:i2 
	%t133 = hw.constant 1 : i1 
	%t117 = SpecHLS.delay %t133 -> %t24 by 6:i32 
	%t134 = hw.constant 1 : i1 
	%t116 = SpecHLS.delay %t134 -> %t55 by 5:i1 
	%t135 = hw.constant 1 : i1 
	%t120 = SpecHLS.delay %t135 -> %t53 by 6:i32 
	%t136 = hw.constant 1 : i1 
	%t119 = SpecHLS.delay %t136 -> %t55 by 5:i1 
	%t137 = hw.constant 1 : i1 
	%t123 = SpecHLS.delay %t137 -> %t65 by 6:i1 
	%t138 = hw.constant 1 : i1 
	%t122 = SpecHLS.delay %t138 -> %t55 by 5:i1 
	hw.output   %t64 : i1 
}
}

