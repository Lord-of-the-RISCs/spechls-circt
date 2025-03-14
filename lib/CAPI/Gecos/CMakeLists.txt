add_library(gecosapi SHARED
    Gecos.cpp
)
target_link_libraries(gecosapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    CIRCT
    MLIR
)

install(TARGETS gecosapi)
set_property(TARGET gecosapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
