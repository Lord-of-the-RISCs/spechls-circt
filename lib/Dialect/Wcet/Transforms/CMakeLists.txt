add_circt_dialect_library(WcetTransforms
    UnrollInstr.cpp
    LongestPath.cpp
    MergeDelays.cpp

    DEPENDS
    WcetTransformsIncGen
)
