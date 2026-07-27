add_library(gecosapi SHARED
    Gecos.cpp
)

if(${SPECHLS_LINK_DYLIB})
  set(LIBS MLIR CIRCT)
else()
  set(LIBS
    MLIRIR
    CIRCTHW
    CIRCTCAPIComb
    CIRCTCAPIHW
    CIRCTCAPIDebug
    CIRCTCAPIFIRRTL
    CIRCTCAPIFSM
    CIRCTCAPISynth
    CIRCTCAPISim
    CIRCTCAPIVerif
    CIRCTCAPISV
    CIRCTCAPIArc
    CIRCTCAPILTL
    CIRCTCAPILLHD
    CIRCTCAPIMoore
    CIRCTCAPISeq
    CIRCTCAPISSP
    CIRCTCAPIHWArith
    CIRCTImportVerilog
    CIRCTExportVerilog
    CIRCTMooreToCore
    CIRCTLLHDTransforms
    CIRCTMooreTransforms
  )
endif()

target_link_libraries(gecosapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    ScheduleToSSP
    SpecHLSTransforms
    SpecHLSToHW
    SpecHLSAnalysis
    MLIR-C
    SpecHLS
    SpecHLSToHW
    Schedule
    ScheduleTransforms

    ${LIBS}

    Yosys::Yosys
)

install(TARGETS gecosapi)
set_property(TARGET gecosapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
