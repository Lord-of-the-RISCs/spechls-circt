add_library(gecosapi SHARED
    Gecos.cpp
)
target_link_libraries(gecosapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    ScheduleToSSP
    SpecHLSTransforms
    SpecHLSToHW
    SpecHLSAnalysis
    CIRCTTransforms
    CIRCTScheduling
    CIRCTCAPIComb
    CIRCTCAPIDebug
    CIRCTCAPIHW
    CIRCTCAPILLHD
    CIRCTCAPILTL
    CIRCTCAPIMoore
    CIRCTCAPISeq
    CIRCTCAPISim
    CIRCTCAPISV
    CIRCTCAPISynth
    CIRCTCAPIVerif
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
