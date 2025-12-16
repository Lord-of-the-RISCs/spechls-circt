spechls.kernel @kernel(%in1 : i32, %in2 : i1) -> i32 {
  spechls.exit if %in2 with %in1 : i32
}

spechls.kernel @kernel2(%in1 : i1, %in2 : i32, %in3 : i32) -> i32 {
  %g = spechls.gamma<"x">(%in1, %in2, %in3) : i1, i32
    %m = spechls.mu<"x">(%in3, %in2) : i32
  %d0 = spechls.delay %in2 by 1 : i32
  spechls.exit if %in1 with %g : i32
}

spechls.kernel @kernel2(%in1 : i1, %in2 : i32, %in3 : i32) -> i32 {
  %g = spechls.gamma<"x">(%in1, %in2, %in3) : i1, i32
    %m = spechls.mu<"x">(%in3, %in2) : i32
//  %d0 = spechls.delay %in2 by 1 : i32
  spechls.exit if %in1 with %g : i32
}

spechls.kernel @kernel3(%array: !spechls.array<i32, 4>, %index: i32, %value: i32, %enable: i1) -> !spechls.array<i32, 4> {

  %result = spechls.alpha %array[%index: i32], %value if %enable : !spechls.array<i32, 4>
  spechls.exit if %enable with %result : !spechls.array<i32, 4>
}
