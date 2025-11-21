add_library(gecosapi SHARED
    Gecos.cpp
)
target_link_libraries(gecosapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    ScheduleToSSP
    SpecHLSTransforms
    SpecHLSToHW
    CIRCT
    MLIR
    MLIR-C
    SpecHLS
    SpecHLSToHW
    Schedule
    ScheduleTransforms

    Yosys::Yosys
)

install(TARGETS gecosapi)
set_property(TARGET gecosapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
