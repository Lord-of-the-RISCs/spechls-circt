add_executable(spechls-opt spechls-opt.cpp)

target_link_libraries(spechls-opt PRIVATE
    SpecHLS
    Schedule
    ScheduleTransforms
    SPECHLSscheduleToSSP
    MLIR
    CIRCT
)
install(TARGETS spechls-opt DESTINATION bin)
set_property(TARGET spechls-opt PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
