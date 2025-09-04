add_mlir_pdll_library(SpecHLSSimplifyGammaNodesPatternsIncGen
    SimplifyGammaNodes.pdll
    SimplifyGammaNodes.h.inc
)

add_mlir_pdll_library(SpecHLSMergeLUTsPatternsIncGen
    MergeLUTs.pdll
    MergeLUTs.h.inc
)

add_circt_dialect_library(SpecHLSTransforms
    MergeLUTs.cpp
    SimplifyGammaNodes.cpp

    DEPENDS
    SpecHLSTransformsIncGen
    SpecHLSMergeLUTsPatternsIncGen
    SpecHLSSimplifyGammaNodesPatternsIncGen
)
target_include_directories(SpecHLSTransforms
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
