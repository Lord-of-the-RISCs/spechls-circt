add_executable(spechls-reduce spechls-reduce.cpp)

target_link_libraries(spechls-reduce PRIVATE
    SpecHLS
    Schedule
    MLIR
    CIRCT
)
install(TARGETS spechls-reduce DESTINATION bin)
set_property(TARGET spechls-reduce PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
