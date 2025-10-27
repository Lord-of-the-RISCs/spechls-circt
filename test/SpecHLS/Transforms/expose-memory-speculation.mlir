

spechls.kernel @simple(%arr : !spechls.array<i32, 16>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 1} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

spechls.kernel @simpleD3(%arr : !spechls.array<i32, 16>, %idxRead : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 3} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}

spechls.kernel @twoRead(%arr : !spechls.array<i32, 16>, %idxRead1 : i32, %idxRead2 : i32, %idxWrite : i32, %valWrite : i32, %we : i1) -> i32 {
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 1} : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %mu[%idxWrite : i32], %valWrite if %we : !spechls.array<i32, 16>
    %operand1 = spechls.load %mu[%idxRead1 : i32] : !spechls.array<i32, 16>
    %operand2 = spechls.load %mu[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}

spechls.kernel @twoWrite(
    %arr : !spechls.array<i32, 16>, %idxRead : i32,
    %idxWrite1 : i32, %idxWrite2 : i32,
    %valWrite1 : i32, %valWrite2 : i32,
    %we1 : i1, %we2 : i1) -> i32 
{
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 1} : !spechls.array<i32, 16>
    %temp_arr = spechls.alpha %mu[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %temp_arr[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    %result = spechls.load %mu[%idxRead : i32] : !spechls.array<i32, 16>
    spechls.exit if %true with %result : i32
}


spechls.kernel @twoReadsTwoWritesD3(
    %arr : !spechls.array<i32, 16>,
    %idxRead1 : i32, %idxRead2 : i32,
    %idxWrite1 : i32, %idxWrite2 : i32,
    %valWrite1 : i32, %valWrite2 : i32,
    %we1 : i1, %we2 : i1) -> i32 
{
    %true = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%arr, %next_arr) {dependenciesDistances = 3} : !spechls.array<i32, 16>
    %temp_arr = spechls.alpha %mu[%idxWrite1 : i32], %valWrite1 if %we1 : !spechls.array<i32, 16>
    %next_arr = spechls.alpha %temp_arr[%idxWrite2 : i32], %valWrite2 if %we2 : !spechls.array<i32, 16>
    %operand1 = spechls.load %mu[%idxRead1 : i32] : !spechls.array<i32, 16>
    %operand2 = spechls.load %mu[%idxRead2 : i32] : !spechls.array<i32, 16>
    %result = comb.add %operand1, %operand2 : i32
    spechls.exit if %true with %result : i32
}
