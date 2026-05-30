#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

enum class CoreType { SYSTOLIC_OS, SYSTOLIC_WS };

enum class DramType { SIMPLE, RAMULATOR1, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

struct CoreConfig {
  CoreType core_type;
  uint32_t core_width;
  uint32_t core_height;

  /* Vector config*/
  uint32_t vector_process_bit;
  uint32_t layernorm_latency = 1;
  uint32_t softmax_latency = 1;
  uint32_t add_latency = 1;
  uint32_t mul_latency = 1;
  uint32_t mac_latency = 1;
  uint32_t div_latency = 1;
  uint32_t exp_latency = 1;
  uint32_t gelu_latency = 1;
  uint32_t add_tree_latency = 1;
  uint32_t scalar_sqrt_latency = 1;
  uint32_t scalar_add_latency = 1;
  uint32_t scalar_mul_latency = 1;

  /* SRAM config */
  uint32_t sram_width;
  uint32_t spad_size;
  uint32_t accum_spad_size;
};

struct SimulationConfig {
  /* Core config */
  uint32_t num_cores;
  uint32_t core_freq;
  uint32_t core_print_interval;
  struct CoreConfig *core_config;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_freq;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  uint32_t dram_size; // in GB
  uint32_t dram_nbl = 1; // busrt length in clock cycles (bust_length 8 in DDR -> 4 nbl)
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /* ICNT config */
  IcntType icnt_type;
  uint32_t icnt_injection_ports_per_core = 1;
  std::string icnt_config_path;
  uint32_t icnt_freq;
  uint32_t icnt_latency;
  uint32_t icnt_print_interval=0;

  /* Sheduler config */
  std::string scheduler_type;

  /* Other configs */
  uint32_t precision;
  uint32_t full_precision = 4;
  std::string layout;

  /* Graph-Bit precision-depth execution.
   *
   * These knobs let a GemmWS tile execute only the high-order activation
   * bit-planes that are required by the graph-risk selected precision depth.
   * full_depth is normally 8 for W4A8, while precision_depth can be 8/6/4.
   */
  bool graphbit_enable = false;
  bool graphbit_bound_enable = false;
  uint32_t graphbit_full_depth = 8;
  uint32_t graphbit_precision_depth = 8;
  uint32_t graphbit_min_depth = 4;
  float graphbit_bound_tolerance = 0.0f;
  float graphbit_bound_scale = 1.0f;
  float graphbit_memory_scale = 1.0f;
  std::string graphbit_activation_layout = "plane_group";
  uint32_t graphbit_plane_group_bits = 1;
  bool graphbit_issue_gate = true;
  bool graphbit_weight_rf_gate = false;
  bool graphbit_psum_gate = false;
  bool graphbit_risk_bucket_enable = true;
  bool graphbit_weight_stationary_enable = false;
  uint32_t graphbit_baseline_weight_tile_batch = 1;
  uint32_t graphbit_weight_stationary_tile_batch = 1;
  float graphbit_weight_memory_scale = 1.0f;

  /*
   * This map stores the partition information: <partition_id, core_id>
   *
   * Note: Each core belongs to one partition. Through these partition IDs,
   * it is possible to assign a specific DNN model to a particular group of cores.
   */
  std::map<uint32_t, std::vector<uint32_t>> partiton_map;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }

  float max_systolic_flops(uint32_t id) {
    return core_config[id].core_width * core_config[id].core_height * core_freq * 2 * num_cores / 1000; // GFLOPS
  }

  float max_vector_flops(uint32_t id) {
    return (core_config[id].vector_process_bit >> 3) / precision * 2 * core_freq / 1000; // GFLOPS
  }

  float max_dram_bandwidth() {
    return dram_freq * dram_channels * dram_req_size / dram_nbl / 1000; // GB/s
  }

};
