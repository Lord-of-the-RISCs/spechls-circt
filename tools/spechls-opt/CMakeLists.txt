add_executable(spechls-opt spechls-opt.cpp)

target_link_libraries(spechls-opt PRIVATE
    SpecHLS
    SpecHLSTransforms
    SpecHLSToHW
    Schedule
    Wcet
    ScheduleTransforms
    WcetTransforms
    ScheduleToSSP
    CIRCTTransforms
    CIRCTScheduling
    MLIR

    Yosys::Yosys
)
install(TARGETS spechls-opt DESTINATION bin)
set_property(TARGET spechls-opt PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
