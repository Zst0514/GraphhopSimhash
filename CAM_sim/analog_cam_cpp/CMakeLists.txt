add_library(analog_cam_engine
    analog_cam_engine.cpp
)
target_link_libraries(analog_cam_engine PUBLIC ghhw_common)

add_executable(analog_cam_reuse
    analog_cam_sim.cpp
)
target_link_libraries(analog_cam_reuse PRIVATE analog_cam_engine ghhw_common)

add_executable(analog_cam_frontend_probe
    analog_cam_frontend_probe.cpp
)
target_link_libraries(analog_cam_frontend_probe PRIVATE analog_cam_engine ghhw_common)
