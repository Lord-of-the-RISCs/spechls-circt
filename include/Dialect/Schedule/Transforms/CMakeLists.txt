set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Schedule)
add_public_tablegen_target(ScheduleTransformsIncGen)
add_circt_doc(Passes SchedulePasses -gen-pass-doc)
