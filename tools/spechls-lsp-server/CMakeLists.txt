add_llvm_executable(spechls-lsp-server SpecHLSLspServerMain.cpp)

llvm_update_compile_flags(spechls-lsp-server)

target_link_libraries(spechls-lsp-server PRIVATE
    MLIRSpecHLS
    SpecHLSScheduleDialect
    CIRCT
    Yosys::Yosys
)
target_compile_options(spechls-lsp-server PUBLIC -fexceptions)
install(TARGETS spechls-lsp-server DESTINATION bin)
