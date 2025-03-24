module{
hw.module.extern @target( in %ir : i32,  in %base : i32 , out "out0" : i32 )

hw.module.extern @mult( in %a : i32,  in %b : i32 , out "out0" : i32 )

hw.module.extern @alu( in %fun : i8,  in %a : i32,  in %b : i32 , out "out0" : i32 )

hw.module.extern @memory_load( in %fun : i8,  in %base : i32,  in %offset : i32 , out "out0" : i32 )

hw.module.extern @taken( in %fun : i8,  in %rs1 : i32,  in %rs2 : i32 , out "out0" : i1 )

hw.module.extern @encoder_4( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1,  in %in_3 : i1 , out "out0" : ui2 )

hw.module.extern @encoder_8( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1,  in %in_3 : i1,  in %in_4 : i1,  in %in_5 : i1,  in %in_6 : i1,  in %in_7 : i1 , out "out0" : ui3 )

hw.module.extern @bv3toint( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1 , out "out0" : ui3 )

hw.module.extern @bv4toint( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1,  in %in_3 : i1 , out "out0" : ui4 )

hw.module @SpecSCC_21(out "out_0": i1) attributes {"#pragma" = "UNROLL_NODE"}{
	%t1 = hw.constant 20 : i32 
	%t2 = hw.constant 1048575 : i32 
	%t3 = hw.constant 12 : i32 
	%t4 = comb.shl %t2,%t3 : i32 
	%t5 = hw.constant 1 : i32 
	%t6 = hw.constant 11 : i32 
	%t7 = comb.shl %t5,%t6 : i32 
	%t8 = hw.constant 255 : i32 
	%t9 = comb.shl %t8,%t3 : i32 
	%t10 = comb.shl %t5,%t1 : i32 
	%t11 = comb.xor %t5,%t5: i32 
	%t12 = SpecHLS.init @regs : memref<32xi32> 
	%t13 = SpecHLS.mu @regs : %t12,%t14 : memref<32xi32> 
	%t15 = SpecHLS.init @mem : memref<256xi32> 
	%t16 = SpecHLS.mu @mem : %t15,%t17 : memref<256xi32> 
	%t18 = SpecHLS.init @pc : i32 
	%t19 = SpecHLS.mu @pc : %t18,%t20 : i32 
	%t21 = SpecHLS.init @isHalted : i1 
	%t22 = SpecHLS.mu @isHalted : %t21,%t23 : i1 
	%t24_idx = arith.index_cast %t19 : i32 to index
	%t24 = SpecHLS.read %t16:memref<256xi32>  [%t24_idx]  {"#pragma" = "entry_point"}
	%t25 = hw.constant 127 : i32 
	%t26 = comb.and %t24,%t25 : i32 
	%t27 = hw.constant 15 : i32 
	%t28 = comb.shrs %t24,%t27 : i32 
	%t29 = hw.constant 31 : i32 
	%t30 = comb.and %t28,%t29 : i32 
	%t31_idx = arith.index_cast %t30 : i32 to index
	%t31 = SpecHLS.read %t13:memref<32xi32>  [%t31_idx]   
	%t32 = hw.constant 7 : i32 
	%t33 = comb.shrs %t24,%t32 : i32 
	%t34 = comb.and %t33,%t29 : i32 
	%t35 = hw.constant 4095 : i32 
	%t36 = hw.constant 4064 : i32 
	%t37 = comb.and %t24,%t4 : i32 
	%t38 = hw.constant 1023 : i32 
	%t39 = hw.constant 9 : i32 
	%t40 = comb.shrs %t24,%t39 : i32 
	%t41 = comb.and %t40,%t7 : i32 
	%t42 = comb.and %t24,%t9 : i32 
	%t43 = hw.constant 19 : i32 
	%t26_c = builtin.unrealized_conversion_cast %t26 : i32 to i32					
	%t43_c = builtin.unrealized_conversion_cast %t43 : i32 to i32					
	%t44 = comb.icmp eq %t26_c, %t43_c : i32
	%t45 = hw.constant 25 : i32 
	%t46 = comb.shrs %t24,%t45 : i32 
	%t47 = hw.constant 0 : i32 
	%t46_c = builtin.unrealized_conversion_cast %t46 : i32 to i32					
	%t47_c = builtin.unrealized_conversion_cast %t47 : i32 to i32					
	%t48 = comb.icmp eq %t46_c, %t47_c : i32
	%t49 = hw.constant 51 : i32 
	%t49_c = builtin.unrealized_conversion_cast %t49 : i32 to i32					
	%t50 = comb.icmp eq %t26_c, %t49_c : i32
	%t51 = hw.constant 3 : i32 
	%t51_c = builtin.unrealized_conversion_cast %t51 : i32 to i32					
	%t52 = comb.icmp eq %t26_c, %t51_c : i32
	%t53 = hw.constant 35 : i32 
	%t53_c = builtin.unrealized_conversion_cast %t53 : i32 to i32					
	%t54 = comb.icmp eq %t26_c, %t53_c : i32
	%t24_c = SpecHLS.cast %t24:i32  to i32
	
	%t19_c = SpecHLS.cast %t19:i32  to i32
					%t55 = hw.instance "%50" @target  ( ir :  %t24_c : i32,  base :  %t19_c : i32 ) -> ( out0 : i32 ) 
	%t56 = hw.constant 99 : i32 
	%t56_c = builtin.unrealized_conversion_cast %t56 : i32 to i32					
	%t57 = comb.icmp eq %t26_c, %t56_c : i32
	%t58 = hw.constant 55 : i32 
	%t58_c = builtin.unrealized_conversion_cast %t58 : i32 to i32					
	%t59 = comb.icmp eq %t26_c, %t58_c : i32
	%t60 = comb.add %t19,%t37 : i32 
	%t61 = hw.constant 23 : i32 
	%t61_c = builtin.unrealized_conversion_cast %t61 : i32 to i32					
	%t62 = comb.icmp eq %t26_c, %t61_c : i32
	%t63 = hw.constant 111 : i32 
	%t63_c = builtin.unrealized_conversion_cast %t63 : i32 to i32					
	%t64 = comb.icmp eq %t26_c, %t63_c : i32
	
	%t31_c = SpecHLS.cast %t31:i32  to i32
					%t65 = hw.instance "%60" @target  ( ir :  %t24_c : i32,  base :  %t31_c : i32 ) -> ( out0 : i32 ) 
	%t66 = comb.and %t65,%t11 : i32 
	%t67 = hw.constant 103 : i32 
	%t67_c = builtin.unrealized_conversion_cast %t67 : i32 to i32					
	%t68 = comb.icmp eq %t26_c, %t67_c : i32
	%t69 = hw.constant 115 : i32 
	%t69_c = builtin.unrealized_conversion_cast %t69 : i32 to i32					
	%t70 = comb.icmp eq %t26_c, %t69_c : i32
	%t71 = hw.constant 1 : i1 
	%t72 = hw.constant 0 : i1 
	%t73 = SpecHLS.init @guard : i1 
	%t74 = SpecHLS.mu @guard : %t73,%t75 : i1 
	%t76 = arith.andi %t54,%t74 : i1 
	%t77 = arith.ori  %t44,%t50 : i1 
	%t78_0 = hw.constant 0 : i1
	%t78 = comb.icmp eq %t78_0,%t77: i1
	%t79 = arith.andi %t50,%t48 : i1 
	%t80_0 = hw.constant 0 : i1
	%t80 = comb.icmp eq %t80_0,%t48: i1
	%t81 = arith.andi %t50,%t80 : i1 
	%t82 = arith.ori  %t79,%t81 : i1 
	%t83 = arith.ori  %t44,%t82 : i1 
	%t84 = arith.ori  %t83,%t52 : i1 
	%t85 = arith.ori  %t84,%t54 : i1 
	%t86 = arith.ori  %t85,%t59 : i1 
	%t87 = arith.ori  %t86,%t62 : i1 
	%t88 = arith.ori  %t70,%t78 : i1 
	%t23 = SpecHLS.gamma @isHalted %t88 ? %t22,%t71 :i1 
	%t89 = arith.ori  %t87,%t88 : i1 
	%t90 = arith.andi %t44,%t74 : i1 
	%t91 = arith.andi %t52,%t74 : i1 
	%t92 = arith.andi %t59,%t74 : i1 
	%t93 = arith.andi %t62,%t74 : i1 
	%t94 = arith.andi %t64,%t74 : i1 
	%t95 = arith.andi %t68,%t74 : i1 
	%t96_0 = hw.constant 0 : i1
	%t96 = comb.icmp eq %t96_0,%t81: i1
	%t97_0 = hw.constant 0 : i1
	%t97 = comb.icmp eq %t97_0,%t23: i1
	%t98 = arith.andi %t74,%t97 : i1 
	%t75 = SpecHLS.gamma @guard %t98 ? %t72,%t71 :i1 
	%t99 = arith.andi %t79,%t74 : i1 
	%t100 = arith.andi %t81,%t74 : i1 
	%t101 = hw.constant 20 : i32 
	%t102 = comb.shrs %t24,%t101 : i32 
	%t103 = comb.and %t102,%t29 : i32 
	%t104_idx = arith.index_cast %t103 : i32 to index
	%t104 = SpecHLS.read %t13:memref<32xi32>  [%t104_idx]   
	%t105 = comb.and %t102,%t35 : i32 
	%t106 = comb.and %t102,%t36 : i32 
	%t107 = comb.add %t34,%t106 : i32 
	%t108 = comb.and %t102,%t38 : i32 
	%t109 = comb.or  %t108,%t41 : i32 
	%t110 = comb.or  %t109,%t42 : i32 
	
	%t104_c = SpecHLS.cast %t104:i32  to i32
					%t111 = hw.instance "%107" @mult  ( a :  %t31_c : i32,  b :  %t104_c : i32 ) -> ( out0 : i32 ) 
	%t112 = comb.add %t31,%t107 : i32 
	%t113 = hw.constant 11 : i32 
	%t114 = comb.shrs %t24,%t113 : i32 
	%t115 = comb.and %t114,%t10 : i32 
	%t116 = comb.or  %t110,%t115 : i32 
	%t117 = comb.add %t19,%t116 : i32 
	%t118 = hw.constant 12 : i32 
	%t119 = comb.shrs %t24,%t118 : i32 
	%t120 = comb.and %t119,%t32 : i32 
	%t120_c = SpecHLS.cast %t120:i32  to i8
	
	
	%t105_c = SpecHLS.cast %t105:i32  to i32
					%t121 = hw.instance "%117" @alu  ( fun :  %t120_c : i8,  a :  %t31_c : i32,  b :  %t105_c : i32 ) -> ( out0 : i32 ) 
	
	
					%t122 = hw.instance "%118" @alu  ( fun :  %t120_c : i8,  a :  %t31_c : i32,  b :  %t104_c : i32 ) -> ( out0 : i32 ) 
	
	
					%t123 = hw.instance "%119" @memory_load  ( fun :  %t120_c : i8,  base :  %t31_c : i32,  offset :  %t105_c : i32 ) -> ( out0 : i32 ) 
	
	
					%t124 = hw.instance "%120" @taken  ( fun :  %t120_c : i8,  rs1 :  %t31_c : i32,  rs2 :  %t104_c : i32 ) -> ( out0 : i1 ) 
	%t125 = arith.andi %t57,%t124 : i1 
	%t126_0 = hw.constant 0 : i1
	%t126 = comb.icmp eq %t126_0,%t124: i1
	%t127 = arith.andi %t57,%t126 : i1 
	%t128 = arith.ori  %t125,%t127 : i1 
	%t89_c = SpecHLS.cast %t89:i1  to i1
	
	%t128_c = SpecHLS.cast %t128:i1  to i1
	
	%t64_c = SpecHLS.cast %t64:i1  to i1
	
	%t68_c = SpecHLS.cast %t68:i1  to i1
  %t129_c0 = hw.constant 0 : i2
  %t129_c1 = hw.constant 1 : i2
  %t129_c2 = hw.constant 2 : i2
  %t129_c3 = hw.constant 3 : i2
  %t129_m0 = comb.mux %t64, %t129_c2, %t129_c3 : i2
  %t129_m1 = comb.mux %t128, %t129_c1, %t129_m0 : i2
  %t129 = comb.mux %t89, %t129_c0, %t129_m1 : i2
	%t130 = arith.ori  %t54,%t128 : i1 
	%t131 = arith.ori  %t130,%t88 : i1 
	
  %t132_c0 = hw.constant 0 : i3
  %t132_c1 = hw.constant 1 : i3
  %t132_c2 = hw.constant 2 : i3
  %t132_c3 = hw.constant 3 : i3
  %t132_c4 = hw.constant 4 : i3
  %t132_c5 = hw.constant 5 : i3
  %t132_c6 = hw.constant 6 : i3
  %t132_c7 = hw.constant 7 : i3
  %t132_m0 = comb.mux %t64, %t132_c6, %t132_c7 : i3
  %t132_m1 = comb.mux %t62, %t132_c5, %t132_m0 : i3
  %t132_m2 = comb.mux %t59, %t132_c4, %t132_m1 : i3
  %t132_m3 = comb.mux %t52, %t132_c3, %t132_m2 : i3
  %t132_m4 = comb.mux %t82, %t132_c2, %t132_m3 : i3
  %t132_m5 = comb.mux %t44, %t132_c1, %t132_m4 : i3
  %t132 = comb.mux %t131, %t132_c0, %t132_m5 : i3
	%t133 = comb.extract %t129 from 0 : (i2)->i1 
	%t134 = comb.extract %t129 from 1 : (i2)->i1 
	%t135_0 = hw.constant 0 : i1
	%t135 = comb.icmp eq %t135_0,%t125: i1
	%t136_0 = hw.constant 0 : i1
	%t136 = comb.icmp eq %t136_0,%t134: i1
	%t137 = arith.andi %t125,%t136 : i1 
	%t138 = arith.andi %t133,%t137 : i1 
	%t139 = arith.andi %t133,%t134 : i1 
	%t140_0 = hw.constant 0 : i1
	%t140 = comb.icmp eq %t140_0,%t139: i1
	%t141 = arith.andi %t135,%t133 : i1 
	%t142 = arith.ori  %t134,%t141 : i1 
	%t143 = arith.andi %t140,%t142 : i1 
	%t144 = arith.ori  %t139,%t141 : i1 
	%t143_c = SpecHLS.cast %t143:i1  to i1
	
	
	%t138_c = SpecHLS.cast %t138:i1  to i1
  %t145 = comb.concat %t138, %t144, %t143 : i1, i1, i1
	%t146 = comb.extract %t132 from 0 : (i3)->i1 
	%t147 = comb.extract %t132 from 1 : (i3)->i1 
	%t148 = comb.extract %t132 from 2 : (i3)->i1 
	%t149_0 = hw.constant 0 : i1
	%t149 = comb.icmp eq %t149_0,%t147: i1
	%t150 = arith.ori  %t148,%t146 : i1 
	%t151_0 = hw.constant 0 : i1
	%t151 = comb.icmp eq %t151_0,%t150: i1
	%t152 = arith.andi %t147,%t151 : i1 
	%t153 = arith.andi %t81,%t152 : i1 
	%t154 = arith.andi %t96,%t152 : i1 
	%t155 = arith.andi %t146,%t149 : i1 
	%t156 = arith.ori  %t148,%t155 : i1 
	%t157 = arith.andi %t148,%t146 : i1 
	%t158_0 = hw.constant 0 : i1
	%t158 = comb.icmp eq %t158_0,%t157: i1
	%t159 = arith.andi %t156,%t158 : i1 
	%t160 = arith.ori  %t154,%t159 : i1 
	%t161 = arith.ori  %t146,%t147 : i1 
	%t162 = comb.xor %t146,%t149 : i1 
	%t163 = arith.andi %t150,%t162 : i1 
	%t164 = arith.ori  %t154,%t163 : i1 
	%t165 = arith.andi %t148,%t161 : i1 
	%t166 = arith.ori  %t154,%t165 : i1 
  %t167 = comb.concat %t153, %t166, %t164, %t160 : i1, i1, i1, i1
	%t168 = SpecHLS.gamma @merge__1 %t169 ? %t47,%t34,%t34,%t34,%t34,%t34,%t34,%t34,%t34 :i32 
	%t170 = SpecHLS.gamma @merge__0 %t171 ? %t72,%t90,%t91,%t92,%t93,%t94,%t95,%t99,%t100 :i1 
	%t172 = hw.constant 1 : i32 
	%t173 = comb.add %t19,%t172 : i32 
	%t174 = SpecHLS.gamma @merge__2 %t175 ? %t47,%t176,%t177,%t37,%t60,%t173,%t173,%t178,%t179 :i32 
	%t14_idx = arith.index_cast %t168 : i32 to index
	%t14 = SpecHLS.alpha @regs : %t170 -> %t13 [%t14_idx], %t174: memref<32xi32>  
	%t20 = SpecHLS.gamma @nextpc %t180 ? %t173,%t181,%t182,%t173,%t183 :i32 
	%t184 = SpecHLS.gamma @writeValue_mem %t185 ? %t47,%t186 :i32 
	%t187 = SpecHLS.gamma @writeAdress_mem %t188 ? %t47,%t189 :i32 
	%t190 = SpecHLS.gamma @writeEnable_mem %t191 ? %t72,%t192 :i1 
	%t17_idx = arith.index_cast %t187 : i32 to index
	%t17 = SpecHLS.alpha @mem : %t190 -> %t16 [%t17_idx], %t184: memref<256xi32>  
	%t193 = hw.constant 1 : i1 
	%t169 = SpecHLS.delay %t193 -> %t167 by 2:i4 
	%t194 = hw.constant 1 : i1 
	%t171 = SpecHLS.delay %t194 -> %t167 by 2:i4 
	%t195 = hw.constant 1 : i1 
	%t176 = SpecHLS.delay %t195 -> %t121 by 3:i32 
	%t196 = hw.constant 1 : i1 
	%t177 = SpecHLS.delay %t196 -> %t123 by 6:i32 
	%t197 = hw.constant 1 : i1 
	%t178 = SpecHLS.delay %t197 -> %t122 by 3:i32 
	%t198 = hw.constant 1 : i1 
	%t179 = SpecHLS.delay %t198 -> %t111 by 6:i32 
	%t199 = hw.constant 1 : i1 
	%t175 = SpecHLS.delay %t199 -> %t167 by 2:i4 
	%t200 = hw.constant 1 : i1 
	%t181 = SpecHLS.delay %t200 -> %t117 by 7:i32 
	%t201 = hw.constant 1 : i1 
	%t182 = SpecHLS.delay %t201 -> %t66 by 7:i32 
	%t202 = hw.constant 1 : i1 
	%t183 = SpecHLS.delay %t202 -> %t55 by 7:i32 
	%t203 = hw.constant 1 : i1 
	%t180 = SpecHLS.delay %t203 -> %t145 by 6:i3 
	%t204 = hw.constant 1 : i1 
	%t186 = SpecHLS.delay %t204 -> %t104 by 6:i32 
	%t205 = hw.constant 1 : i1 
	%t185 = SpecHLS.delay %t205 -> %t54 by 5:i1 
	%t206 = hw.constant 1 : i1 
	%t189 = SpecHLS.delay %t206 -> %t112 by 6:i32 
	%t207 = hw.constant 1 : i1 
	%t188 = SpecHLS.delay %t207 -> %t54 by 5:i1 
	%t208 = hw.constant 1 : i1 
	%t192 = SpecHLS.delay %t208 -> %t76 by 6:i1 
	%t209 = hw.constant 1 : i1 
	%t191 = SpecHLS.delay %t209 -> %t54 by 5:i1 
	hw.output   %t75 : i1 
}
}
