set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SPECHLSConversion)
add_public_tablegen_target(SPECHLSConversionPassIncGen)