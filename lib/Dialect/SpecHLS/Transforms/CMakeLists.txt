add_mlir_pdll_library(SpecHLSPDLLPatternsIncGen
    SimplifyGammaNodes.pdll
    SimplifyGammaNodes.h.inc
)

add_circt_dialect_library(SpecHLSTransforms
    SimplifyGammaNodes.cpp

    DEPENDS
    SpecHLSTransformsIncGen
    SpecHLSPDLLPatternsIncGen
)
target_include_directories(SpecHLSTransforms
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
