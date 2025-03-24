module{
  hw.module.extern @target( in %ir : i32,  in %base : i32 , out "out0" : i32 )

  hw.module.extern @mult( in %a : i32,  in %b : i32 , out "out0" : i32 )

  hw.module.extern @alu( in %fun : i8,  in %a : i32,  in %b : i32 , out "out0" : i32 )

  hw.module.extern @memory_load( in %fun : i8,  in %base : i32,  in %offset : i32 , out "out0" : i32 )

  hw.module.extern @taken( in %fun : i8,  in %rs1 : i32,  in %rs2 : i32 , out "out0" : i1 )

  hw.module @encoder_4( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1,  in %in_3 : i1 , out "out0" : i2 ) attributes {"#pragma" = "INLINE"}
  {
    %c_0 = hw.constant 0 : i2
    %c_1 = hw.constant 1 : i2
    %c_2 = hw.constant 2 : i2
    %c_3 = hw.constant 3 : i2
    
    %tmp0 = comb.mux %in_0, %c_0, %c_0 : i2
    %tmp1 = comb.mux %in_1, %c_1, %tmp0 : i2
    %tmp2 = comb.mux %in_2, %c_2, %tmp1 : i2
    %tmp3 = comb.mux %in_3, %c_3, %tmp2 : i2

    hw.output %tmp3 : i2
  }

  hw.module @g_ctrl__0( in %g0control : i2,  in %g4control : i1 , out "out0" : i3 ) attributes {"#pragma" = "INLINE"}
  {
    %c0 = hw.constant 0 : i3
    %c1 = hw.constant 1 : i3
    %c2 = hw.constant 2 : i3
    %c3 = hw.constant 3 : i3
    %c4 = hw.constant 4 : i3

    %g00 = hw.constant 0 : i2
    %g01 = hw.constant 1 : i2
    %g02 = hw.constant 2 : i2
    %g03 = hw.constant 3 : i2

    %eq00 = comb.icmp eq %g0control, %g00 : i2
    %eq01 = comb.icmp eq %g0control, %g01 : i2
    %eq02 = comb.icmp eq %g0control, %g02 : i2
    %eq03 = comb.icmp eq %g0control, %g03 : i2
    
    %g40 = hw.constant 0 : i1
    %g41 = hw.constant 1 : i1
    
    %eq40 = comb.icmp eq %g4control, %g40 : i1
    %eq41 = comb.icmp eq %g4control, %g41 : i1
    
    %and0 = comb.and %eq01, %eq40 : i1
    %and1 = comb.and %eq01, %eq41 : i1

    %tmp0 = comb.mux %eq00, %c0, %c0 : i3
    %tmp1 = comb.mux %eq02, %c1, %tmp0 : i3
    %tmp2 = comb.mux %eq03, %c2, %tmp1 : i3
    %tmp3 = comb.mux %and0, %c3, %tmp2 : i3
    %tmp4 = comb.mux %and1, %c4, %tmp2 : i3

    hw.output %tmp4 : i3
  }

  hw.module @encoder_8( in %in_0 : i1,  in %in_1 : i1,  in %in_2 : i1,  in %in_3 : i1,  in %in_4 : i1,  in %in_5 : i1,  in %in_6 : i1,  in %in_7 : i1 , out "out0" : i3 ) attributes {"#pragma" = "INLINE"}
  {
    %c_0 = hw.constant 0 : i3
    %c_1 = hw.constant 1 : i3
    %c_2 = hw.constant 2 : i3
    %c_3 = hw.constant 3 : i3
    %c_4 = hw.constant 4 : i3
    %c_5 = hw.constant 5 : i3
    %c_6 = hw.constant 6 : i3
    %c_7 = hw.constant 7 : i3
    
    %tmp0 = comb.mux %in_0, %c_0, %c_0 : i3
    %tmp1 = comb.mux %in_1, %c_1, %tmp0 : i3
    %tmp2 = comb.mux %in_2, %c_2, %tmp1 : i3
    %tmp3 = comb.mux %in_3, %c_3, %tmp2 : i3
    %tmp4 = comb.mux %in_4, %c_4, %tmp3 : i3
    %tmp5 = comb.mux %in_5, %c_5, %tmp4 : i3
    %tmp6 = comb.mux %in_6, %c_6, %tmp5 : i3
    %tmp7 = comb.mux %in_7, %c_7, %tmp6 : i3

    hw.output %tmp7 : i3
  }

  hw.module @g_ctrl__2( in %g0control : i3,  in %g8control : i1 , out "out0" : i4 ) attributes {"#pragma" = "INLINE"}
  {
    %c0 = hw.constant 0 : i4
    %c1 = hw.constant 1 : i4
    %c2 = hw.constant 2 : i4
    %c3 = hw.constant 3 : i4
    %c4 = hw.constant 4 : i4
    %c5 = hw.constant 5 : i4
    %c6 = hw.constant 6 : i4
    %c7 = hw.constant 7 : i4
    %c8 = hw.constant 8 : i4

    %g00 = hw.constant 0 : i3
    %g01 = hw.constant 1 : i3
    %g02 = hw.constant 2 : i3
    %g03 = hw.constant 3 : i3
    %g04 = hw.constant 4 : i3
    %g05 = hw.constant 5 : i3
    %g06 = hw.constant 6 : i3
    %g07 = hw.constant 7 : i3

    %eq00 = comb.icmp eq %g0control, %g00 : i3
    %eq01 = comb.icmp eq %g0control, %g01 : i3
    %eq02 = comb.icmp eq %g0control, %g02 : i3
    %eq03 = comb.icmp eq %g0control, %g03 : i3
    %eq04 = comb.icmp eq %g0control, %g04 : i3
    %eq05 = comb.icmp eq %g0control, %g05 : i3
    %eq06 = comb.icmp eq %g0control, %g06 : i3
    %eq07 = comb.icmp eq %g0control, %g07 : i3
    
    %g80 = hw.constant 0 : i1
    %g81 = hw.constant 1 : i1
    
    %eq40 = comb.icmp eq %g8control, %g80 : i1
    %eq41 = comb.icmp eq %g8control, %g81 : i1
    
    %and0 = comb.and %eq02, %eq40 : i1
    %and1 = comb.and %eq02, %eq41 : i1

    %tmp0 = comb.mux %eq00, %c0, %c0 : i4
    %tmp1 = comb.mux %eq01, %c1, %tmp0 : i4
    %tmp2 = comb.mux %eq03, %c2, %tmp1 : i4
    %tmp3 = comb.mux %eq04, %c3, %tmp2 : i4
    %tmp4 = comb.mux %eq05, %c4, %tmp3 : i4
    %tmp5 = comb.mux %eq06, %c5, %tmp4 : i4
    %tmp6 = comb.mux %eq07, %c6, %tmp5 : i4
    %tmp7 = comb.mux %and0, %c7, %tmp6 : i4
    %tmp8 = comb.mux %and1, %c8, %tmp7 : i4

    hw.output %tmp8 : i4
  }

  hw.module @g_ctrl__3( in %g0control : i3,  in %g8control : i1 , out "out0" : i4 ) attributes {"#pragma" = "INLINE"}
  {
    %c0 = hw.constant 0 : i4
    %c1 = hw.constant 1 : i4
    %c2 = hw.constant 2 : i4
    %c3 = hw.constant 3 : i4
    %c4 = hw.constant 4 : i4
    %c5 = hw.constant 5 : i4
    %c6 = hw.constant 6 : i4
    %c7 = hw.constant 7 : i4
    %c8 = hw.constant 8 : i4

    %g00 = hw.constant 0 : i3
    %g01 = hw.constant 1 : i3
    %g02 = hw.constant 2 : i3
    %g03 = hw.constant 3 : i3
    %g04 = hw.constant 4 : i3
    %g05 = hw.constant 5 : i3
    %g06 = hw.constant 6 : i3
    %g07 = hw.constant 7 : i3

    %eq00 = comb.icmp eq %g0control, %g00 : i3
    %eq01 = comb.icmp eq %g0control, %g01 : i3
    %eq02 = comb.icmp eq %g0control, %g02 : i3
    %eq03 = comb.icmp eq %g0control, %g03 : i3
    %eq04 = comb.icmp eq %g0control, %g04 : i3
    %eq05 = comb.icmp eq %g0control, %g05 : i3
    %eq06 = comb.icmp eq %g0control, %g06 : i3
    %eq07 = comb.icmp eq %g0control, %g07 : i3
    
    %g80 = hw.constant 0 : i1
    %g81 = hw.constant 1 : i1
    
    %eq40 = comb.icmp eq %g8control, %g80 : i1
    %eq41 = comb.icmp eq %g8control, %g81 : i1
    
    %and0 = comb.and %eq02, %eq40 : i1
    %and1 = comb.and %eq02, %eq41 : i1

    %tmp0 = comb.mux %eq00, %c0, %c0 : i4
    %tmp1 = comb.mux %eq01, %c1, %tmp0 : i4
    %tmp2 = comb.mux %eq03, %c2, %tmp1 : i4
    %tmp3 = comb.mux %eq04, %c3, %tmp2 : i4
    %tmp4 = comb.mux %eq05, %c4, %tmp3 : i4
    %tmp5 = comb.mux %eq06, %c5, %tmp4 : i4
    %tmp6 = comb.mux %eq07, %c6, %tmp5 : i4
    %tmp7 = comb.mux %and0, %c7, %tmp6 : i4
    %tmp8 = comb.mux %and1, %c8, %tmp7 : i4

    hw.output %tmp8 : i4
  }

  hw.module @g_ctrl__1( in %g0control : i3,  in %g8control : i1 , out "out0" : i4 ) attributes {"#pragma" = "INLINE"}
  {
    %c0 = hw.constant 0 : i4
    %c1 = hw.constant 1 : i4
    %c2 = hw.constant 2 : i4
    %c3 = hw.constant 3 : i4
    %c4 = hw.constant 4 : i4
    %c5 = hw.constant 5 : i4
    %c6 = hw.constant 6 : i4
    %c7 = hw.constant 7 : i4
    %c8 = hw.constant 8 : i4

    %g00 = hw.constant 0 : i3
    %g01 = hw.constant 1 : i3
    %g02 = hw.constant 2 : i3
    %g03 = hw.constant 3 : i3
    %g04 = hw.constant 4 : i3
    %g05 = hw.constant 5 : i3
    %g06 = hw.constant 6 : i3
    %g07 = hw.constant 7 : i3

    %eq00 = comb.icmp eq %g0control, %g00 : i3
    %eq01 = comb.icmp eq %g0control, %g01 : i3
    %eq02 = comb.icmp eq %g0control, %g02 : i3
    %eq03 = comb.icmp eq %g0control, %g03 : i3
    %eq04 = comb.icmp eq %g0control, %g04 : i3
    %eq05 = comb.icmp eq %g0control, %g05 : i3
    %eq06 = comb.icmp eq %g0control, %g06 : i3
    %eq07 = comb.icmp eq %g0control, %g07 : i3
    
    %g80 = hw.constant 0 : i1
    %g81 = hw.constant 1 : i1
    
    %eq40 = comb.icmp eq %g8control, %g80 : i1
    %eq41 = comb.icmp eq %g8control, %g81 : i1
    
    %and0 = comb.and %eq02, %eq40 : i1
    %and1 = comb.and %eq02, %eq41 : i1

    %tmp0 = comb.mux %eq00, %c0, %c0 : i4
    %tmp1 = comb.mux %eq01, %c1, %tmp0 : i4
    %tmp2 = comb.mux %eq03, %c2, %tmp1 : i4
    %tmp3 = comb.mux %eq04, %c3, %tmp2 : i4
    %tmp4 = comb.mux %eq05, %c4, %tmp3 : i4
    %tmp5 = comb.mux %eq06, %c5, %tmp4 : i4
    %tmp6 = comb.mux %eq07, %c6, %tmp5 : i4
    %tmp7 = comb.mux %and0, %c7, %tmp6 : i4
    %tmp8 = comb.mux %and1, %c8, %tmp7 : i4

    hw.output %tmp8 : i4
  }

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
    %t24 = SpecHLS.read %t16:memref<256xi32>  [%t24_idx] {"#pragma" = "entry_point"} 
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
    %t44 = comb.icmp eq %t26, %t43 : i32
    %t45 = hw.constant 25 : i32 
    %t46 = comb.shrs %t24,%t45 : i32 
    %t47 = hw.constant 0 : i32 
    %t48 = comb.icmp eq %t46, %t47 : i32
    %t49 = hw.constant 51 : i32 
    %t50 = comb.icmp eq %t26, %t49 : i32
    %t51 = hw.constant 3 : i32 
    %t52 = comb.icmp eq %t26, %t51 : i32
    %t53 = hw.constant 35 : i32 
    %t54 = comb.icmp eq %t26, %t53 : i32
    
    %t55 = hw.instance "%50" @target  ( ir :  %t24 : i32,  base :  %t19 : i32 ) -> ( out0 : i32 ) 
    %t56 = hw.constant 99 : i32 
    %t57 = comb.icmp eq %t26, %t56 : i32
    %t58 = hw.constant 55 : i32 
    %t59 = comb.icmp eq %t26, %t58 : i32
    %t60 = comb.add %t19,%t37 : i32 
    %t61 = hw.constant 23 : i32 
    %t62 = comb.icmp eq %t26, %t61 : i32
    %t63 = hw.constant 111 : i32 
    %t64 = comb.icmp eq %t26, %t63 : i32
    %t65 = hw.instance "%60" @target  ( ir :  %t24 : i32,  base :  %t31 : i32 ) -> ( out0 : i32 ) 
    %t66 = comb.and %t65,%t11 : i32 
    %t67 = hw.constant 103 : i32 
    %t68 = comb.icmp eq %t26, %t67 : i32
    %t69 = hw.constant 115 : i32 
    %t70 = comb.icmp eq %t26, %t69 : i32
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
    %t87_no = comb.icmp eq %t80_0, %t87 : i1
    %t88 = arith.ori  %t70,%t78 : i1 
    %t23 = SpecHLS.gamma @isHalted %t87_no ? %t22,%t71 :i1 
    %t89_0 = hw.constant 0 : i1
    %t89 = comb.icmp eq %t89_0,%t23: i1
    %t90 = arith.andi %t74,%t89 : i1 
    %t75 = SpecHLS.gamma @guard %t90 ? %t72,%t71 :i1 
    %t91 = arith.ori  %t87,%t88 : i1 
    %t92 = arith.andi %t44,%t74 : i1 
    %t93 = arith.andi %t52,%t74 : i1 
    %t94 = arith.andi %t59,%t74 : i1 
    %t95 = arith.andi %t62,%t74 : i1 
    %t96 = arith.andi %t64,%t74 : i1 
    %t97 = arith.andi %t68,%t74 : i1 
    %t98 = arith.andi %t79,%t74 : i1 
    %t99 = arith.andi %t81,%t74 : i1 
    %t100 = hw.constant 20 : i32 
    %t101 = comb.shrs %t24,%t100 : i32 
    %t102 = comb.and %t101,%t29 : i32 
    %t103_idx = arith.index_cast %t102 : i32 to index
    %t103 = SpecHLS.read %t13:memref<32xi32>  [%t103_idx]   
    %t104 = comb.and %t101,%t35 : i32 
    %t105 = comb.and %t101,%t36 : i32 
    %t106 = comb.add %t34,%t105 : i32 
    %t107 = comb.and %t101,%t38 : i32 
    %t108 = comb.or  %t107,%t41 : i32 
    %t109 = comb.or  %t108,%t42 : i32 
    %t110 = hw.instance "%106" @mult  ( a :  %t31 : i32,  b :  %t103 : i32 ) -> ( out0 : i32 ) 
    %t111 = comb.add %t31,%t106 : i32 
    %t112 = hw.constant 11 : i32 
    %t113 = comb.shrs %t24,%t112 : i32 
    %t114 = comb.and %t113,%t10 : i32 
    %t115 = comb.or  %t109,%t114 : i32 
    %t116 = comb.add %t19,%t115 : i32 
    %t117 = hw.constant 12 : i32 
    %t118 = comb.shrs %t24,%t117 : i32 
    %t119 = comb.and %t118,%t32 : i32 
    %t119_c = SpecHLS.cast %t119:i32  to i8
    %t120 = hw.instance "%116" @alu  ( fun :  %t119_c : i8,  a :  %t31 : i32,  b :  %t104 : i32 ) -> ( out0 : i32 ) 
    %t121 = hw.instance "%117" @alu  ( fun :  %t119_c : i8,  a :  %t31 : i32,  b :  %t103 : i32 ) -> ( out0 : i32 ) 
    %t122 = hw.instance "%118" @memory_load  ( fun :  %t119_c : i8,  base :  %t31 : i32,  offset :  %t104 : i32 ) -> ( out0 : i32 ) 
    %t123 = hw.instance "%119" @taken  ( fun :  %t119_c : i8,  rs1 :  %t31 : i32,  rs2 :  %t103 : i32 ) -> ( out0 : i1 ) 
    %t124 = arith.andi %t57,%t123 : i1 
    %t125_0 = hw.constant 0 : i1
    %t125 = comb.icmp eq %t125_0,%t123: i1
    %t126 = arith.andi %t57,%t125 : i1 
    %t127 = arith.ori  %t124,%t126 : i1 
    %t128 = hw.instance "%124" @encoder_4  ( in_0 :  %t91 : i1,  in_1 :  %t127 : i1,  in_2 :  %t64 : i1,  in_3 :  %t68 : i1 ) -> ( out0 : i2 ) 
    %t129 = hw.instance "%125" @g_ctrl__0  ( g0control :  %t128 : i2,  g4control :  %t124 : i1 ) -> ( out0 : i3 ) 
    %t130 = arith.ori  %t54,%t127 : i1 
    %t131 = arith.ori  %t130,%t88 : i1 
    %t132 = hw.instance "%128" @encoder_8  ( in_0 :  %t131 : i1,  in_1 :  %t44 : i1,  in_2 :  %t82 : i1,  in_3 :  %t52 : i1,  in_4 :  %t59 : i1,  in_5 :  %t62 : i1,  in_6 :  %t64 : i1,  in_7 :  %t68 : i1 ) -> ( out0 : i3 ) 
    %t133 = hw.instance "%129" @g_ctrl__2  ( g0control :  %t132 : i3,  g8control :  %t81 : i1 ) -> ( out0 : i4 ) 
    %t134 = SpecHLS.gamma @merge__1 %t135 ? %t47,%t34,%t34,%t34,%t34,%t34,%t34,%t34,%t34 :i32 
    %t136 = hw.instance "%131" @g_ctrl__3  ( g0control :  %t132 : i3,  g8control :  %t81 : i1 ) -> ( out0 : i4 ) 
    %t137 = hw.instance "%132" @g_ctrl__1  ( g0control :  %t132 : i3,  g8control :  %t81 : i1 ) -> ( out0 : i4 ) 
    %t138 = SpecHLS.gamma @merge__0 %t139 ?%t72,%t92,%t93,%t94,%t95,%t96,%t97,%t98,%t99 :i1 
    %t140 = hw.constant 1 : i32 
    %t141 = comb.add %t19,%t140 : i32 
    %t142 = SpecHLS.gamma @merge__2 %t143 ? %t47,%t144,%t145,%t37,%t60,%t141,%t141,%t146,%t147 :i32 
    %t14_idx = arith.index_cast %t134 : i32 to index
    %t14 = SpecHLS.alpha @regs : %t138 -> %t13 [%t14_idx], %t142: memref<32xi32>  
    %t20 = SpecHLS.gamma @nextpc %t148 ? %t141,%t149,%t150,%t141,%t151 :i32 
    %t152 = SpecHLS.gamma @writeValue_mem %t153 ? %t47,%t154 :i32 
    %t155 = SpecHLS.gamma @writeAdress_mem %t156 ? %t47,%t157 :i32 
    %t158 = SpecHLS.gamma @writeEnable_mem %t159 ? %t72,%t160 :i1 
    %t17_idx = arith.index_cast %t155 : i32 to index
    %t17 = SpecHLS.alpha @mem : %t158 -> %t16 [%t17_idx], %t152: memref<256xi32>  
    %t161 = hw.constant 1 : i1 
    %t135 = SpecHLS.delay %t161 -> %t133 by 2:i4 
    %t162 = hw.constant 1 : i1 
    %t139 = SpecHLS.delay %t162 -> %t137 by 2:i4 
    %t163 = hw.constant 1 : i1 
    %t144 = SpecHLS.delay %t163 -> %t120 by 3:i32 
    %t164 = hw.constant 1 : i1 
    %t145 = SpecHLS.delay %t164 -> %t122 by 6:i32 
    %t165 = hw.constant 1 : i1 
    %t146 = SpecHLS.delay %t165 -> %t121 by 3:i32 
    %t166 = hw.constant 1 : i1 
    %t147 = SpecHLS.delay %t166 -> %t110 by 6:i32 
    %t167 = hw.constant 1 : i1 
    %t143 = SpecHLS.delay %t167 -> %t136 by 2:i4 
    %t168 = hw.constant 1 : i1 
    %t149 = SpecHLS.delay %t168 -> %t116 by 7:i32 
    %t169 = hw.constant 1 : i1 
    %t150 = SpecHLS.delay %t169 -> %t66 by 7:i32 
    %t170 = hw.constant 1 : i1 
    %t151 = SpecHLS.delay %t170 -> %t55 by 7:i32 
    %t171 = hw.constant 1 : i1 
    %t148 = SpecHLS.delay %t171 -> %t129 by 6:i3 
    %t172 = hw.constant 1 : i1 
    %t154 = SpecHLS.delay %t172 -> %t103 by 6:i32 
    %t173 = hw.constant 1 : i1 
    %t153 = SpecHLS.delay %t173 -> %t54 by 5:i1 
    %t174 = hw.constant 1 : i1 
    %t157 = SpecHLS.delay %t174 -> %t111 by 6:i32 
    %t175 = hw.constant 1 : i1 
    %t156 = SpecHLS.delay %t175 -> %t54 by 5:i1 
    %t176 = hw.constant 1 : i1 
    %t160 = SpecHLS.delay %t176 -> %t76 by 6:i1 
    %t177 = hw.constant 1 : i1 
    %t159 = SpecHLS.delay %t177 -> %t54 by 5:i1 
    hw.output   %t75 : i1 
  }
}
