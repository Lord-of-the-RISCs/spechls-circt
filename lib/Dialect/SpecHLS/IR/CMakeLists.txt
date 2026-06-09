add_circt_dialect_library(SpecHLS
    SpecHLS.cpp
    Inlining.cpp

    DEPENDS
    MLIRSpecHLSIncGen
)
