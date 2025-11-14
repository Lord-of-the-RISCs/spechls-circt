add_mlir_pdll_library(WcetSplitDelaysPatternsIncGen
  SplitDelays.pdll
  SplitDelays.h.inc
)


add_circt_dialect_library(WcetTransforms
    UnrollInstr.cpp
    LongestPath.cpp
    SplitDelays.cpp
    WcetAnalysis.cpp
    InlineTasks.cpp
    InsertDummy.cpp

    DEPENDS
    WcetTransformsIncGen
    WcetSplitDelaysPatternsIncGen
)
target_include_directories(WcetTransforms
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
