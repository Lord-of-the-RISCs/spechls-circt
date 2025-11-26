add_mlir_pdll_library(WcetSplitDelaysPatternsIncGen
  SplitDelays.pdll
  SplitDelays.h.inc
)

add_mlir_pdll_library(WcetInsertDummyPatternsIncGen
  InsertDummy.pdll
  InsertDummy.h.inc
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
    UnrollInstr.cpp
    LongestPath.cpp
    SplitDelays.cpp
    WcetAnalysis.cpp
    InlineCore.cpp
    InsertDummy.cpp
    OutlineCore.cpp
    LUTToMux.cpp
    GammaBinarization.cpp

    DEPENDS
    WcetTransformsIncGen
    WcetSplitDelaysPatternsIncGen
    WcetInsertDummyPatternsIncGen
    WcetLUTToMuxPatternsIncGen
    WcetGammaBinarizationPatternsIncGen
)
target_include_directories(WcetTransforms
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
