module {
  hw.module.extern @target(in %ir : i32, in %base : i32, out out0 : i32)
  hw.module.extern @mult(in %a : i32, in %b : i32, out out0 : i32)
  hw.module.extern @alu(in %fun : i8, in %a : i32, in %b : i32, out out0 : i32)
  hw.module.extern @memory_load(in %fun : i8, in %base : i32, in %offset : i32, out out0 : i32)
  hw.module.extern @taken(in %fun : i8, in %rs1 : i32, in %rs2 : i32, out out0 : i1)
  hw.module @SpecSCC_21(out out_0 : i1) attributes {"#pragma" = "UNROLL_NODE"} {
    %c20_i32 = hw.constant 20 : i32
    %c1048575_i32 = hw.constant 1048575 : i32
    %c12_i32 = hw.constant 12 : i32
    %0 = comb.shl %c1048575_i32, %c12_i32 : i32
    %c1_i32 = hw.constant 1 : i32
    %c11_i32 = hw.constant 11 : i32
    %1 = comb.shl %c1_i32, %c11_i32 : i32
    %c255_i32 = hw.constant 255 : i32
    %2 = comb.shl %c255_i32, %c12_i32 : i32
    %3 = comb.shl %c1_i32, %c20_i32 : i32
    %4 = comb.xor %c1_i32, %c1_i32 : i32
    %5 = SpecHLS.init @regs : memref<32xi32>
    %mu = SpecHLS.mu @regs : %5, %alpha : memref<32xi32>
    %6 = SpecHLS.init @mem : memref<256xi32>
    %mu_0 = SpecHLS.mu @mem : %6, %alpha_87 : memref<256xi32>
    %7 = SpecHLS.init @pc : i32
    %mu_1 = SpecHLS.mu @pc : %7, %gamma_83 : i32
    %8 = SpecHLS.init @isHalted : i1
    %mu_2 = SpecHLS.mu @isHalted : %8, %gamma : i1
    %9 = arith.index_cast %mu_1 : i32 to index
    %10 = SpecHLS.read %mu_0 : memref<256xi32>[%9] {"#pragma" = "entry_point"}
    %c127_i32 = hw.constant 127 : i32
    %11 = comb.and %10, %c127_i32 : i32
    %c15_i32 = hw.constant 15 : i32
    %12 = comb.shrs %10, %c15_i32 : i32
    %c31_i32 = hw.constant 31 : i32
    %13 = comb.and %12, %c31_i32 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15 = SpecHLS.read %mu : memref<32xi32>[%14]
    %c7_i32 = hw.constant 7 : i32
    %16 = comb.shrs %10, %c7_i32 : i32
    %17 = comb.and %16, %c31_i32 : i32
    %c4095_i32 = hw.constant 4095 : i32
    %c4064_i32 = hw.constant 4064 : i32
    %18 = comb.and %10, %0 : i32
    %c1023_i32 = hw.constant 1023 : i32
    %c9_i32 = hw.constant 9 : i32
    %19 = comb.shrs %10, %c9_i32 : i32
    %20 = comb.and %19, %1 : i32
    %21 = comb.and %10, %2 : i32
    %c19_i32 = hw.constant 19 : i32
    %22 = builtin.unrealized_conversion_cast %11 : i32 to i32
    %23 = builtin.unrealized_conversion_cast %c19_i32 : i32 to i32
    %24 = comb.icmp eq %22, %23 : i32
    %c25_i32 = hw.constant 25 : i32
    %25 = comb.shrs %10, %c25_i32 : i32
    %c0_i32 = hw.constant 0 : i32
    %26 = builtin.unrealized_conversion_cast %25 : i32 to i32
    %27 = builtin.unrealized_conversion_cast %c0_i32 : i32 to i32
    %28 = comb.icmp eq %26, %27 : i32
    %c51_i32 = hw.constant 51 : i32
    %29 = builtin.unrealized_conversion_cast %c51_i32 : i32 to i32
    %30 = comb.icmp eq %22, %29 : i32
    %c3_i32 = hw.constant 3 : i32
    %31 = builtin.unrealized_conversion_cast %c3_i32 : i32 to i32
    %32 = comb.icmp eq %22, %31 : i32
    %c35_i32 = hw.constant 35 : i32
    %33 = builtin.unrealized_conversion_cast %c35_i32 : i32 to i32
    %34 = comb.icmp eq %22, %33 : i32
    %35 = SpecHLS.cast %10 : i32 to i32
    %36 = SpecHLS.cast %mu_1 : i32 to i32
    %2550 = hw.instance "%50" @target(ir: %35: i32, base: %36: i32) -> (out0: i32)
    %c99_i32 = hw.constant 99 : i32
    %37 = builtin.unrealized_conversion_cast %c99_i32 : i32 to i32
    %38 = comb.icmp eq %22, %37 : i32
    %c55_i32 = hw.constant 55 : i32
    %39 = builtin.unrealized_conversion_cast %c55_i32 : i32 to i32
    %40 = comb.icmp eq %22, %39 : i32
    %41 = comb.add %mu_1, %18 : i32
    %c23_i32 = hw.constant 23 : i32
    %42 = builtin.unrealized_conversion_cast %c23_i32 : i32 to i32
    %43 = comb.icmp eq %22, %42 : i32
    %c111_i32 = hw.constant 111 : i32
    %44 = builtin.unrealized_conversion_cast %c111_i32 : i32 to i32
    %45 = comb.icmp eq %22, %44 : i32
    %46 = SpecHLS.cast %15 : i32 to i32
    %2560 = hw.instance "%60" @target(ir: %35: i32, base: %46: i32) -> (out0: i32)
    %47 = comb.and %2560, %4 : i32
    %c103_i32 = hw.constant 103 : i32
    %48 = builtin.unrealized_conversion_cast %c103_i32 : i32 to i32
    %49 = comb.icmp eq %22, %48 : i32
    %c115_i32 = hw.constant 115 : i32
    %50 = builtin.unrealized_conversion_cast %c115_i32 : i32 to i32
    %51 = comb.icmp eq %22, %50 : i32
    %true = hw.constant true
    %false = hw.constant false
    %52 = SpecHLS.init @guard : i1
    %mu_3 = SpecHLS.mu @guard : %52, %gamma_7 : i1
    %53 = arith.andi %34, %mu_3 : i1
    %54 = arith.ori %24, %30 : i1
    %false_4 = hw.constant false
    %55 = comb.icmp eq %false_4, %54 : i1
    %56 = arith.andi %30, %28 : i1
    %false_5 = hw.constant false
    %57 = comb.icmp eq %false_5, %28 : i1
    %58 = arith.andi %30, %57 : i1
    %59 = arith.ori %56, %58 : i1
    %60 = arith.ori %24, %59 : i1
    %61 = arith.ori %60, %32 : i1
    %62 = arith.ori %61, %34 : i1
    %63 = arith.ori %62, %40 : i1
    %64 = arith.ori %63, %43 : i1
    %65 = arith.ori %51, %55 : i1
    %gamma = SpecHLS.gamma @isHalted %65:i1 ? %mu_2,%true :i1
    %false_6 = hw.constant false
    %66 = comb.icmp eq %false_6, %gamma : i1
    %67 = arith.andi %mu_3, %66 : i1
    %gamma_7 = SpecHLS.gamma @guard %67:i1 ? %false,%true :i1
    %68 = arith.ori %64, %65 : i1
    %69 = arith.andi %24, %mu_3 : i1
    %70 = arith.andi %32, %mu_3 : i1
    %71 = arith.andi %40, %mu_3 : i1
    %72 = arith.andi %43, %mu_3 : i1
    %73 = arith.andi %45, %mu_3 : i1
    %74 = arith.andi %49, %mu_3 : i1
    %75 = arith.andi %56, %mu_3 : i1
    %76 = arith.andi %58, %mu_3 : i1
    %c20_i32_8 = hw.constant 20 : i32
    %77 = comb.shrs %10, %c20_i32_8 : i32
    %78 = comb.and %77, %c31_i32 : i32
    %79 = arith.index_cast %78 : i32 to index
    %80 = SpecHLS.read %mu : memref<32xi32>[%79]
    %81 = comb.and %77, %c4095_i32 : i32
    %82 = comb.and %77, %c4064_i32 : i32
    %83 = comb.add %17, %82 : i32
    %84 = comb.and %77, %c1023_i32 : i32
    %85 = comb.or %84, %20 : i32
    %86 = comb.or %85, %21 : i32
    %87 = SpecHLS.cast %80 : i32 to i32
    %25106 = hw.instance "%106" @mult(a: %46: i32, b: %87: i32) -> (out0: i32)
    %88 = comb.add %15, %83 : i32
    %c11_i32_9 = hw.constant 11 : i32
    %89 = comb.shrs %10, %c11_i32_9 : i32
    %90 = comb.and %89, %3 : i32
    %91 = comb.or %86, %90 : i32
    %92 = comb.add %mu_1, %91 : i32
    %c12_i32_10 = hw.constant 12 : i32
    %93 = comb.shrs %10, %c12_i32_10 : i32
    %94 = comb.and %93, %c7_i32 : i32
    %95 = SpecHLS.cast %94 : i32 to i8
    %96 = SpecHLS.cast %81 : i32 to i32
    %25116 = hw.instance "%116" @alu(fun: %95: i8, a: %46: i32, b: %96: i32) -> (out0: i32)
    %25117 = hw.instance "%117" @alu(fun: %95: i8, a: %46: i32, b: %87: i32) -> (out0: i32)
    %25118 = hw.instance "%118" @memory_load(fun: %95: i8, base: %46: i32, offset: %96: i32) -> (out0: i32)
    %25119 = hw.instance "%119" @taken(fun: %95: i8, rs1: %46: i32, rs2: %87: i32) -> (out0: i1)
    %97 = arith.andi %38, %25119 : i1
    %false_11 = hw.constant false
    %98 = comb.icmp eq %false_11, %25119 : i1
    %99 = arith.andi %38, %98 : i1
    %100 = arith.ori %97, %99 : i1
    %101 = SpecHLS.cast %68 : i1 to i1
    %102 = SpecHLS.cast %100 : i1 to i1
    %103 = SpecHLS.cast %45 : i1 to i1
    %104 = SpecHLS.cast %49 : i1 to i1
    %c0_i2 = hw.constant 0 : i2
    %c1_i2 = hw.constant 1 : i2
    %c-2_i2 = hw.constant -2 : i2
    %c-1_i2 = hw.constant -1 : i2
    %105 = comb.mux %101, %c0_i2, %c-1_i2 : i2
    %106 = comb.mux %102, %c1_i2, %105 : i2
    %107 = comb.mux %103, %c-2_i2, %106 : i2
    %108 = comb.mux %104, %c-1_i2, %107 : i2
    %c0_i3 = hw.constant 0 : i3
    %c0_i3_12 = hw.constant 0 : i3
    %c0_i2_13 = hw.constant 0 : i2
    %c1_i2_14 = hw.constant 1 : i2
    %c-2_i2_15 = hw.constant -2 : i2
    %c-1_i2_16 = hw.constant -1 : i2
    %c0_i3_17 = hw.constant 0 : i3
    %c1_i3 = hw.constant 1 : i3
    %c2_i3 = hw.constant 2 : i3
    %c3_i3 = hw.constant 3 : i3
    %c-4_i3 = hw.constant -4 : i3
    %109 = comb.icmp eq %108, %c0_i2_13 : i2
    %110 = comb.icmp eq %108, %c1_i2_14 : i2
    %111 = comb.icmp eq %108, %c-2_i2_15 : i2
    %112 = comb.icmp eq %108, %c-1_i2_16 : i2
    %false_18 = hw.constant false
    %true_19 = hw.constant true
    %113 = comb.icmp eq %97, %false_18 : i1
    %114 = comb.icmp eq %97, %true_19 : i1
    %115 = comb.mux %109, %c0_i3, %c0_i3_12 : i3
    %116 = comb.mux %111, %c1_i3, %115 : i3
    %117 = comb.mux %112, %c2_i3, %116 : i3
    %118 = comb.and %110, %113 : i1
    %119 = comb.and %110, %114 : i1
    %120 = comb.mux %118, %c3_i3, %117 : i3
    %121 = comb.mux %119, %c-4_i3, %120 : i3
    %122 = arith.ori %34, %100 : i1
    %123 = arith.ori %122, %65 : i1
    %124 = SpecHLS.cast %24 : i1 to i1
    %125 = SpecHLS.cast %59 : i1 to i1
    %126 = SpecHLS.cast %32 : i1 to i1
    %127 = SpecHLS.cast %40 : i1 to i1
    %128 = SpecHLS.cast %43 : i1 to i1
    %c0_i3_20 = hw.constant 0 : i3
    %c1_i3_21 = hw.constant 1 : i3
    %c2_i3_22 = hw.constant 2 : i3
    %c3_i3_23 = hw.constant 3 : i3
    %c-4_i3_24 = hw.constant -4 : i3
    %c-3_i3 = hw.constant -3 : i3
    %c-2_i3 = hw.constant -2 : i3
    %c-1_i3 = hw.constant -1 : i3
    %129 = comb.mux %123, %c0_i3_20, %c-1_i3 : i3
    %130 = comb.mux %24, %c1_i3_21, %129 : i3
    %131 = comb.mux %59, %c2_i3_22, %130 : i3
    %132 = comb.mux %32, %c3_i3_23, %131 : i3
    %133 = comb.mux %123, %c-4_i3_24, %132 : i3
    %134 = comb.mux %24, %c-3_i3, %133 : i3
    %135 = comb.mux %59, %c-2_i3, %134 : i3
    %136 = comb.mux %32, %c-1_i3, %135 : i3
    %137 = SpecHLS.cast %58 : i1 to ui1
    %c0_i4 = hw.constant 0 : i4
    %c0_i4_25 = hw.constant 0 : i4
    %c0_i3_26 = hw.constant 0 : i3
    %c1_i3_27 = hw.constant 1 : i3
    %c2_i3_28 = hw.constant 2 : i3
    %c3_i3_29 = hw.constant 3 : i3
    %c-4_i3_30 = hw.constant -4 : i3
    %c-3_i3_31 = hw.constant -3 : i3
    %c-2_i3_32 = hw.constant -2 : i3
    %c-1_i3_33 = hw.constant -1 : i3
    %c0_i4_34 = hw.constant 0 : i4
    %c1_i4 = hw.constant 1 : i4
    %c2_i4 = hw.constant 2 : i4
    %c3_i4 = hw.constant 3 : i4
    %c4_i4 = hw.constant 4 : i4
    %c5_i4 = hw.constant 5 : i4
    %c6_i4 = hw.constant 6 : i4
    %c7_i4 = hw.constant 7 : i4
    %c-8_i4 = hw.constant -8 : i4
    %138 = comb.icmp eq %136, %c0_i3_26 : i3
    %139 = comb.icmp eq %136, %c1_i3_27 : i3
    %140 = comb.icmp eq %136, %c2_i3_28 : i3
    %141 = comb.icmp eq %136, %c3_i3_29 : i3
    %142 = comb.icmp eq %136, %c-4_i3_30 : i3
    %143 = comb.icmp eq %136, %c-3_i3_31 : i3
    %144 = comb.icmp eq %136, %c-2_i3_32 : i3
    %145 = comb.icmp eq %136, %c-1_i3_33 : i3
    %false_35 = hw.constant false
    %true_36 = hw.constant true
    %146 = comb.icmp eq %58, %false_35 : i1
    %147 = comb.icmp eq %58, %true_36 : i1
    %148 = comb.mux %138, %c0_i4, %c0_i4_25 : i4
    %149 = comb.mux %139, %c1_i4, %148 : i4
    %150 = comb.mux %141, %c3_i4, %149 : i4
    %151 = comb.mux %142, %c4_i4, %150 : i4
    %152 = comb.mux %143, %c5_i4, %151 : i4
    %153 = comb.mux %144, %c6_i4, %152 : i4
    %154 = comb.mux %145, %c7_i4, %153 : i4
    %155 = comb.and %140, %146 : i1
    %156 = comb.and %140, %147 : i1
    %157 = comb.mux %155, %c7_i4, %154 : i4
    %158 = comb.mux %156, %c-8_i4, %157 : i4
    %gamma_37 = SpecHLS.gamma @merge__1 %204:i4 ? %c0_i32,%17,%17,%17,%17,%17,%17,%17,%17 :i32
    %c0_i4_38 = hw.constant 0 : i4
    %c0_i4_39 = hw.constant 0 : i4
    %c0_i3_40 = hw.constant 0 : i3
    %c1_i3_41 = hw.constant 1 : i3
    %c2_i3_42 = hw.constant 2 : i3
    %c3_i3_43 = hw.constant 3 : i3
    %c-4_i3_44 = hw.constant -4 : i3
    %c-3_i3_45 = hw.constant -3 : i3
    %c-2_i3_46 = hw.constant -2 : i3
    %c-1_i3_47 = hw.constant -1 : i3
    %c0_i4_48 = hw.constant 0 : i4
    %c1_i4_49 = hw.constant 1 : i4
    %c2_i4_50 = hw.constant 2 : i4
    %c3_i4_51 = hw.constant 3 : i4
    %c4_i4_52 = hw.constant 4 : i4
    %c5_i4_53 = hw.constant 5 : i4
    %c6_i4_54 = hw.constant 6 : i4
    %c7_i4_55 = hw.constant 7 : i4
    %c-8_i4_56 = hw.constant -8 : i4
    %159 = comb.icmp eq %136, %c0_i3_40 : i3
    %160 = comb.icmp eq %136, %c1_i3_41 : i3
    %161 = comb.icmp eq %136, %c2_i3_42 : i3
    %162 = comb.icmp eq %136, %c3_i3_43 : i3
    %163 = comb.icmp eq %136, %c-4_i3_44 : i3
    %164 = comb.icmp eq %136, %c-3_i3_45 : i3
    %165 = comb.icmp eq %136, %c-2_i3_46 : i3
    %166 = comb.icmp eq %136, %c-1_i3_47 : i3
    %false_57 = hw.constant false
    %true_58 = hw.constant true
    %167 = comb.icmp eq %58, %false_57 : i1
    %168 = comb.icmp eq %58, %true_58 : i1
    %169 = comb.mux %159, %c0_i4_38, %c0_i4_39 : i4
    %170 = comb.mux %160, %c1_i4_49, %169 : i4
    %171 = comb.mux %162, %c3_i4_51, %170 : i4
    %172 = comb.mux %163, %c4_i4_52, %171 : i4
    %173 = comb.mux %164, %c5_i4_53, %172 : i4
    %174 = comb.mux %165, %c6_i4_54, %173 : i4
    %175 = comb.mux %166, %c7_i4_55, %174 : i4
    %176 = comb.and %161, %167 : i1
    %177 = comb.and %161, %168 : i1
    %178 = comb.mux %176, %c7_i4_55, %175 : i4
    %179 = comb.mux %177, %c-8_i4_56, %178 : i4
    %c0_i4_59 = hw.constant 0 : i4
    %c0_i4_60 = hw.constant 0 : i4
    %c0_i3_61 = hw.constant 0 : i3
    %c1_i3_62 = hw.constant 1 : i3
    %c2_i3_63 = hw.constant 2 : i3
    %c3_i3_64 = hw.constant 3 : i3
    %c-4_i3_65 = hw.constant -4 : i3
    %c-3_i3_66 = hw.constant -3 : i3
    %c-2_i3_67 = hw.constant -2 : i3
    %c-1_i3_68 = hw.constant -1 : i3
    %c0_i4_69 = hw.constant 0 : i4
    %c1_i4_70 = hw.constant 1 : i4
    %c2_i4_71 = hw.constant 2 : i4
    %c3_i4_72 = hw.constant 3 : i4
    %c4_i4_73 = hw.constant 4 : i4
    %c5_i4_74 = hw.constant 5 : i4
    %c6_i4_75 = hw.constant 6 : i4
    %c7_i4_76 = hw.constant 7 : i4
    %c-8_i4_77 = hw.constant -8 : i4
    %180 = comb.icmp eq %136, %c0_i3_61 : i3
    %181 = comb.icmp eq %136, %c1_i3_62 : i3
    %182 = comb.icmp eq %136, %c2_i3_63 : i3
    %183 = comb.icmp eq %136, %c3_i3_64 : i3
    %184 = comb.icmp eq %136, %c-4_i3_65 : i3
    %185 = comb.icmp eq %136, %c-3_i3_66 : i3
    %186 = comb.icmp eq %136, %c-2_i3_67 : i3
    %187 = comb.icmp eq %136, %c-1_i3_68 : i3
    %false_78 = hw.constant false
    %true_79 = hw.constant true
    %188 = comb.icmp eq %58, %false_78 : i1
    %189 = comb.icmp eq %58, %true_79 : i1
    %190 = comb.mux %180, %c0_i4_59, %c0_i4_60 : i4
    %191 = comb.mux %181, %c1_i4_70, %190 : i4
    %192 = comb.mux %183, %c3_i4_72, %191 : i4
    %193 = comb.mux %184, %c4_i4_73, %192 : i4
    %194 = comb.mux %185, %c5_i4_74, %193 : i4
    %195 = comb.mux %186, %c6_i4_75, %194 : i4
    %196 = comb.mux %187, %c7_i4_76, %195 : i4
    %197 = comb.and %182, %188 : i1
    %198 = comb.and %182, %189 : i1
    %199 = comb.mux %197, %c7_i4_76, %196 : i4
    %200 = comb.mux %198, %c-8_i4_77, %199 : i4
    %gamma_80 = SpecHLS.gamma @merge__0 %205:i4 ? %false,%69,%70,%71,%72,%73,%74,%75,%76 :i1
    %c1_i32_81 = hw.constant 1 : i32
    %201 = comb.add %mu_1, %c1_i32_81 : i32
    %gamma_82 = SpecHLS.gamma @merge__2 %210:i4 ? %c0_i32,%206,%207,%18,%41,%201,%201,%208,%209 :i32
    %202 = arith.index_cast %gamma_37 : i32 to index
    %alpha = SpecHLS.alpha @regs : %gamma_80 -> %mu[%202], %gamma_82 : memref<32xi32>
    %gamma_83 = SpecHLS.gamma @nextpc %214:i3 ? %201,%211,%212,%201,%213 :i32
    %gamma_84 = SpecHLS.gamma @writeValue_mem %216:i1 ? %c0_i32,%215 :i32
    %gamma_85 = SpecHLS.gamma @writeAdress_mem %218:i1 ? %c0_i32,%217 :i32
    %gamma_86 = SpecHLS.gamma @writeEnable_mem %220:i1 ? %false,%219 :i1
    %203 = arith.index_cast %gamma_85 : i32 to index
    %alpha_87 = SpecHLS.alpha @mem : %gamma_86 -> %mu_0[%203], %gamma_84 : memref<256xi32>
    %true_88 = hw.constant true
    %204 = SpecHLS.delay %true_88 -> %158 by 2 : i4
    %true_89 = hw.constant true
    %205 = SpecHLS.delay %true_89 -> %200 by 2 : i4
    %true_90 = hw.constant true
    %206 = SpecHLS.delay %true_90 -> %25116 by 3 : i32
    %true_91 = hw.constant true
    %207 = SpecHLS.delay %true_91 -> %25118 by 6 : i32
    %true_92 = hw.constant true
    %208 = SpecHLS.delay %true_92 -> %25117 by 3 : i32
    %true_93 = hw.constant true
    %209 = SpecHLS.delay %true_93 -> %25106 by 6 : i32
    %true_94 = hw.constant true
    %210 = SpecHLS.delay %true_94 -> %179 by 2 : i4
    %true_95 = hw.constant true
    %211 = SpecHLS.delay %true_95 -> %92 by 7 : i32
    %true_96 = hw.constant true
    %212 = SpecHLS.delay %true_96 -> %47 by 7 : i32
    %true_97 = hw.constant true
    %213 = SpecHLS.delay %true_97 -> %2550 by 7 : i32
    %true_98 = hw.constant true
    %214 = SpecHLS.delay %true_98 -> %121 by 6 : i3
    %true_99 = hw.constant true
    %215 = SpecHLS.delay %true_99 -> %80 by 6 : i32
    %true_100 = hw.constant true
    %216 = SpecHLS.delay %true_100 -> %34 by 5 : i1
    %true_101 = hw.constant true
    %217 = SpecHLS.delay %true_101 -> %88 by 6 : i32
    %true_102 = hw.constant true
    %218 = SpecHLS.delay %true_102 -> %34 by 5 : i1
    %true_103 = hw.constant true
    %219 = SpecHLS.delay %true_103 -> %53 by 6 : i1
    %true_104 = hw.constant true
    %220 = SpecHLS.delay %true_104 -> %34 by 5 : i1
    hw.output %gamma_7 : i1
  }
}

