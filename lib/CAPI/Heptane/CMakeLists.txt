add_library(heptaneapi SHARED
  Heptane.cpp
)
target_link_libraries(heptaneapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    WcetCAPI
    ScheduleToSSP
    SpecHLSTransforms
    SpecHLSToHW
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
    Wcet
    WcetTransforms

    Yosys::Yosys
)

install(TARGETS heptaneapi)
set_property(TARGET heptaneapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
