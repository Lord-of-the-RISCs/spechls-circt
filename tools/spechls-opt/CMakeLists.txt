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
    MLIR
    CIRCT

    Yosys::Yosys
)
install(TARGETS spechls-opt DESTINATION bin)
set_property(TARGET spechls-opt PROPERTY INSTALL_RPATH "$ORIGIN/../lib")

# MLIR's aggregate link interface includes these runtime shared libraries.
# Install them with the tool so its existing $ORIGIN/../lib rpath works.
foreach(MLIR_RUNTIME_TARGET
    mlir_c_runner_utils
    mlir_runner_utils
    mlir_float16_utils
    mlir_apfloat_wrappers
    mlir_async_runtime
    mlir_arm_runner_utils
    mlir_arm_sme_abi_stubs)
  if(TARGET ${MLIR_RUNTIME_TARGET})
    install(FILES "$<TARGET_FILE:${MLIR_RUNTIME_TARGET}>" DESTINATION lib)
  endif()
endforeach()
