add_mlir_public_c_api_library(SpecHLSCAPI
        SpecHLSCAPI.cpp
        SSP.cpp
        LINK_LIBS PUBLIC
        MLIRSpecHLS
        SpecHLSScheduleDialect
        MLIR
)
