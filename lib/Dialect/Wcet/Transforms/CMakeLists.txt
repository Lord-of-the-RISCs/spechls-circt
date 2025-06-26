add_circt_dialect_library(WcetTransforms
    UnrollInstr.cpp
    LongestPath.cpp

    DEPENDS
    WcetTransformsIncGen
)
