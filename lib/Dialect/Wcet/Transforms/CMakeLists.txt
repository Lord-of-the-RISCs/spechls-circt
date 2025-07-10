add_circt_dialect_library(WcetTransforms
    UnrollInstr.cpp
    LongestPath.cpp
    SplitDelays.cpp
    WcetAnalysis.cpp
    InlineTasks.cpp

    DEPENDS
    WcetTransformsIncGen
)
