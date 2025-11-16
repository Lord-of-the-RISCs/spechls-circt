add_circt_dialect(SpecHLS spechls)
# ---- Attributes: generate decls/defs from SpecHLS.td ----
set(LLVM_TARGET_DEFINITIONS SpecHLS.td)
mlir_tablegen(SpecHLSAttributes.h.inc  -gen-attrdef-decls -name SpecHLS)
mlir_tablegen(SpecHLSAttributes.cpp.inc -gen-attrdef-defs  -name SpecHLS)
add_public_tablegen_target(SpecHLSAttributesIncGen)

# Make your library (or circt-headers) depend on the generated files
# so they are built before compiling sources that include them.
add_dependencies(circt-headers SpecHLSAttributesIncGen)
# or: add_dependencies(CIRCTSpecHLS SpecHLSAttributesIncGen)
add_circt_dialect_doc(SpecHLS spechls)
