add_mlir_pdll_library(SpecHLSSimplifyGammasPatternsIncGen
    SimplifyGammas.pdll
    SimplifyGammas.h.inc
)

add_mlir_pdll_library(SpecHLSSimplifyLUTsPatternsIncGen
    SimplifyLUTs.pdll
    SimplifyLUTs.h.inc
)

add_mlir_pdll_library(SpecHLSRemoveIntraRAWPatternsIncGen
  RemoveIntraRAW.pdll
  RemoveIntraRAW.h.inc
)

add_mlir_pdll_library(SpecHLSExposeMemorySpeculationPatternsIncGen
  ExposeMemorySpeculation.pdll
  ExposeMemorySpeculation.h.inc
)

add_circt_dialect_library(SpecHLSTransforms
    Outlining.cpp
    SimplifyLUTs.cpp
    SimplifyGammas.cpp
    OptimizeControlLogic.cpp
    RemoveIntraRAW.cpp
    ExposeMemorySpeculation.cpp

    DEPENDS
    SpecHLSTransformsIncGen
    SpecHLSSimplifyLUTsPatternsIncGen
    SpecHLSSimplifyGammasPatternsIncGen
    SpecHLSRemoveIntraRAWPatternsIncGen
    SpecHLSExposeMemorySpeculationPatternsIncGen
)
target_include_directories(SpecHLSTransforms
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
