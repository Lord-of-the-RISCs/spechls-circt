add_circt_dialect_library(SpecHLS
    SpecHLS.cpp
        ../Common/utils.cpp

    DEPENDS
    MLIRSpecHLSIncGen
        SpecHLSAttributesIncGen
)
