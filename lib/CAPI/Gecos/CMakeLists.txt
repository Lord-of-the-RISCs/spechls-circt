add_library(gecosapi SHARED
    Gecos.cpp
)
target_link_libraries(gecosapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    ScheduleToSSP
    CIRCT
    MLIR
    MLIR-C
)

install(TARGETS gecosapi)
set_property(TARGET gecosapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
