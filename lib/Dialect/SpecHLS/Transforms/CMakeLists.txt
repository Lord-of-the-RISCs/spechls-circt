add_mlir_pdll_library(SpecHLSSimplifyGammasPatternsIncGen
    SimplifyGammas.pdll
    SimplifyGammas.h.inc
)

add_mlir_pdll_library(SpecHLSSimplifyLUTsPatternsIncGen
    SimplifyLUTs.pdll
    SimplifyLUTs.h.inc
)

add_circt_dialect_library(SpecHLSTransforms
    Outlining.cpp
    SimplifyLUTs.cpp
    SimplifyGammas.cpp
    OptimizeControlLogic.cpp

    DEPENDS
    SpecHLSTransformsIncGen
    SpecHLSSimplifyLUTsPatternsIncGen
    SpecHLSSimplifyGammasPatternsIncGen
)
target_include_directories(SpecHLSTransforms
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
