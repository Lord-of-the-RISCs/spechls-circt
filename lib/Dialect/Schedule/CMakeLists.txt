add_mlir_dialect_library(SpecHLSScheduleDialect
        ScheduleDialect.cpp
        ScheduleOps.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../include/Dialect/SpecHLS

        DEPENDS
        MLIRSpecHLSOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)
