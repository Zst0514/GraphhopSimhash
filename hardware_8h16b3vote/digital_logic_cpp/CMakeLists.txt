add_library(digital_engine
    digital_engine.cpp
)
target_link_libraries(digital_engine PUBLIC ghhw_common)

add_executable(digital_hash_reuse
    digital_sim.cpp
)
target_link_libraries(digital_hash_reuse PRIVATE digital_engine ghhw_common)
