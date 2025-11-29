add_mlir_pdll_library(WcetSplitDelaysPatternsIncGen
  SplitDelays.pdll
  SplitDelays.h.inc
)

add_mlir_pdll_library(WcetLUTToMuxPatternsIncGen
  LUTToMux.pdll
  LUTToMux.h.inc
)

add_mlir_pdll_library(WcetGammaBinarizationPatternsIncGen
  GammaBinarization.pdll
  GammaBinarization.h.inc
)

add_circt_dialect_library(WcetTransforms
    InsertInstr.cpp
    LongestPath.cpp
    SplitDelays.cpp
    InlineCore.cpp
    OutlineCore.cpp
    LUTToMux.cpp
    GammaBinarization.cpp
    SetupAnalysis.cpp

    DEPENDS
    WcetTransformsIncGen
    WcetSplitDelaysPatternsIncGen
    WcetLUTToMuxPatternsIncGen
    WcetGammaBinarizationPatternsIncGen
)
target_include_directories(WcetTransforms
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
