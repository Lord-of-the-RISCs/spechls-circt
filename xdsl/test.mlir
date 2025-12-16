// Example code of a simple counter.
// CIRCT example code may not always work out of the box because the textual MLIR format is not always stable.
// The example tries to be compatible with the latest CIRCT version, using relatively stable IR.

hw.module @Counter(in %data: i8, out count: i8) {
  hw.output %data : i8
}
