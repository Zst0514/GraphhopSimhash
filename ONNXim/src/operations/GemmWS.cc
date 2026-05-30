#include "GemmWS.h"

#include "../Model.h"

#include <algorithm>
#include <cmath>

namespace
{

  uint32_t clamp_graphbit_depth(uint32_t depth, const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    uint32_t min_depth = std::max(1u, std::min(config.graphbit_min_depth, full_depth));
    return std::max(min_depth, std::min(depth, full_depth));
  }

  double estimate_graphbit_remaining_bound(uint32_t depth, uint32_t tile_k,
                                           const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    depth = std::min(depth, full_depth);
    if (depth >= full_depth)
    {
      return 0.0;
    }

    double full_range = std::pow(2.0, static_cast<double>(full_depth)) - 1.0;
    double omitted_range =
        std::pow(2.0, static_cast<double>(full_depth - depth)) - 1.0;
    double normalized_omitted = omitted_range / std::max(1.0, full_range);

    // Legacy bound: only the remaining low-bit range and K-tile width are used.
    // This is useful as a lower-fidelity baseline and is kept selectable for
    // ablations against the tile-aware predictor-free bound below.
    if (config.graphbit_bound_mode == "range")
    {
      double k_scale = std::sqrt(static_cast<double>(std::max(1u, tile_k))) /
                       std::sqrt(128.0);
      return normalized_omitted * k_scale * config.graphbit_bound_scale;
    }

    // Tile-aware predictor-free bound.  No oracle embedding or learned
    // predictor is used.  We bound the uncomputed low-bit contribution with
    //
    //   ||A_low @ W|| <= max_abs(A_low) * sum_abs(W_tile)
    //
    // and normalize by a partial-output norm proxy.  ONNXim does not carry
    // runtime activation/weight values, so W statistics are modeled with
    // hardware-visible tile metadata knobs.  The mean mode is the deployable
    // default; max mode is a conservative stress test.
    double tile = static_cast<double>(std::max(1u, tile_k));
    double weight_mean =
        std::max(1.0e-9, static_cast<double>(config.graphbit_bound_weight_abs_mean));
    double weight_max =
        std::max(weight_mean, static_cast<double>(config.graphbit_bound_weight_abs_max));
    double remaining_weight =
        (config.graphbit_bound_mode == "tile_max") ? weight_max : weight_mean;
    double remaining_bound =
        normalized_omitted * tile * remaining_weight *
        std::max(0.0, static_cast<double>(config.graphbit_bound_safety_factor));

    double high_range = std::max(0.0, 1.0 - normalized_omitted);
    double partial_norm =
        high_range * tile * weight_mean *
        std::max(0.0, static_cast<double>(config.graphbit_bound_partial_norm_scale));
    partial_norm =
        std::max(partial_norm,
                 static_cast<double>(config.graphbit_bound_partial_norm_floor));

    double normalized_bound =
        remaining_bound / std::max(1.0e-12, partial_norm + remaining_bound);

    // Keep the old K-width guard as a configurable safety multiplier.  This
    // makes larger K tiles slightly harder to terminate early unless explicitly
    // tuned down.
    double k_scale = std::sqrt(static_cast<double>(std::max(1u, tile_k))) /
                     std::sqrt(128.0);
    return normalized_bound * k_scale * config.graphbit_bound_scale;
  }

  uint32_t select_graphbit_effective_depth(uint32_t tile_k,
                                           const SimulationConfig &config)
  {
    uint32_t config_depth =
        clamp_graphbit_depth(config.graphbit_precision_depth, config);
    if (!config.graphbit_enable || !config.graphbit_bound_enable)
    {
      return config_depth;
    }

    uint32_t min_depth =
        clamp_graphbit_depth(std::min(config.graphbit_min_depth, config_depth),
                             config);
    for (uint32_t depth = min_depth; depth <= config_depth; depth++)
    {
      if (estimate_graphbit_remaining_bound(depth, tile_k, config) <=
          config.graphbit_bound_tolerance)
      {
        return depth;
      }
    }
    return config_depth;
  }

  uint32_t graphbit_grouped_depth(uint32_t depth,
                                  const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    uint32_t group_bits = std::max(1u, config.graphbit_plane_group_bits);
    depth = std::min(depth, full_depth);
    uint32_t grouped = ((depth + group_bits - 1) / group_bits) * group_bits;
    return std::min(grouped, full_depth);
  }

  uint32_t graphbit_issue_depth(uint32_t effective_depth,
                                const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    if (!config.graphbit_enable || !config.graphbit_issue_gate ||
        !config.graphbit_risk_bucket_enable)
    {
      return full_depth;
    }
    return std::max(1u, std::min(effective_depth, full_depth));
  }

  uint32_t graphbit_fetch_depth(uint32_t effective_depth,
                                const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    if (!config.graphbit_enable || !config.graphbit_risk_bucket_enable ||
        config.graphbit_activation_layout == "byte_major")
    {
      return full_depth;
    }
    return graphbit_grouped_depth(effective_depth, config);
  }

  uint32_t graphbit_weight_depth(uint32_t issue_depth,
                                 const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    if (!config.graphbit_enable || !config.graphbit_weight_rf_gate)
    {
      return full_depth;
    }
    return std::max(1u, std::min(issue_depth, full_depth));
  }

  uint32_t graphbit_psum_depth(uint32_t issue_depth,
                               const SimulationConfig &config)
  {
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    if (!config.graphbit_enable || !config.graphbit_psum_gate)
    {
      return full_depth;
    }
    return std::max(1u, std::min(issue_depth, full_depth));
  }

  double graphbit_weight_hbm_scale(const SimulationConfig &config)
  {
    if (!config.graphbit_enable || !config.graphbit_weight_stationary_enable)
    {
      return 1.0;
    }
    double baseline = static_cast<double>(
        std::max(1u, config.graphbit_baseline_weight_tile_batch));
    double stationary = static_cast<double>(
        std::max(1u, config.graphbit_weight_stationary_tile_batch));
    double batch_scale = std::min(1.0, baseline / stationary);
    double explicit_scale =
        std::max(0.0f, std::min(1.0f, config.graphbit_weight_memory_scale));
    return std::max(0.0, std::min(1.0, batch_scale * explicit_scale));
  }

  std::vector<addr_type> make_graphbit_src_addrs(const std::set<addr_type> &src,
                                                 uint32_t fetch_depth,
                                                 const SimulationConfig &config)
  {
    std::vector<addr_type> addrs(src.begin(), src.end());
    if (!config.graphbit_enable || addrs.empty())
    {
      return addrs;
    }
    uint32_t full_depth = std::max(1u, config.graphbit_full_depth);
    fetch_depth = std::min(fetch_depth, full_depth);
    double depth_scale =
        static_cast<double>(fetch_depth) / static_cast<double>(full_depth);
    double memory_scale =
        std::max(0.0f, std::min(1.0f, config.graphbit_memory_scale));
    size_t reduced_size = static_cast<size_t>(
        std::ceil(addrs.size() * depth_scale * memory_scale));
    reduced_size = std::max<size_t>(1, std::min(reduced_size, addrs.size()));
    addrs.resize(reduced_size);
    return addrs;
  }

  std::vector<addr_type> make_graphbit_weight_addrs(const std::set<addr_type> &src,
                                                    const SimulationConfig &config)
  {
    std::vector<addr_type> addrs(src.begin(), src.end());
    if (!config.graphbit_enable || addrs.empty())
    {
      return addrs;
    }
    double scale = graphbit_weight_hbm_scale(config);
    size_t reduced_size =
        static_cast<size_t>(std::ceil(addrs.size() * scale));
    reduced_size = std::max<size_t>(1, std::min(reduced_size, addrs.size()));
    addrs.resize(reduced_size);
    return addrs;
  }

  void annotate_graphbit(Instruction &inst, uint32_t tile_k,
                         const SimulationConfig &config)
  {
    if (!config.graphbit_enable)
    {
      return;
    }
    uint32_t effective_depth = select_graphbit_effective_depth(tile_k, config);
    inst.graphbit_enabled = true;
    inst.graphbit_full_depth = std::max(1u, config.graphbit_full_depth);
    inst.graphbit_config_depth =
        clamp_graphbit_depth(config.graphbit_precision_depth, config);
    inst.graphbit_effective_depth = effective_depth;
    inst.graphbit_fetch_depth = graphbit_fetch_depth(effective_depth, config);
    inst.graphbit_issue_depth = graphbit_issue_depth(effective_depth, config);
    inst.graphbit_weight_depth =
        graphbit_weight_depth(inst.graphbit_issue_depth, config);
    inst.graphbit_psum_depth =
        graphbit_psum_depth(inst.graphbit_issue_depth, config);
    inst.graphbit_remaining_bound =
        estimate_graphbit_remaining_bound(effective_depth, tile_k, config);
    inst.graphbit_weight_hbm_scale = graphbit_weight_hbm_scale(config);
  }

} // namespace

GemmWS::GemmWS(SimulationConfig config, Model *model,
               onnx::NodeProto &node_proto, uint32_t target_core)
    : Gemm(config, model, node_proto, target_core) {}

GemmWS::GemmWS(SimulationConfig config, Model *model, onnx::NodeProto &node_proto, bool has_bias, uint32_t target_core)
    : GemmWS(config, model, node_proto, target_core)
{
  this->has_bias = has_bias;
}

GemmWS::GemmWS(SimulationConfig config, MappingTable &mapping_table,
               std::vector<uint32_t> input_shape,
               std::vector<uint32_t> weight_shape,
               std::vector<uint32_t> output_shape,
               uint32_t target_core)
    : Gemm(config, mapping_table, input_shape, weight_shape, output_shape, target_core) {}

GemmWS::GemmWS(SimulationConfig config, Model *model, std::string name,
               std::map<std::string, std::string> &attributes, uint32_t target_core)
    : Gemm(config, model, name, attributes, target_core)
{
  has_bias = std::stoi(get_attribute("has_bias"));
}

void GemmWS::initialize_tiles(MappingTable &mapping_table)
{
  Mapping::LoopCounts key{.N = _output_shape[_input_shape.size() - 2 + Ndim] * _batch_size,
                          .C = _weight_shape[Cdim_w],
                          .M = _weight_shape[Mdim],
                          .S = 1,
                          .R = 1,
                          .Q = 1,
                          .P = 1,
                          .target_core = target_core};

  Mapping mapping;
  try
  {
    mapping = mapping_table.at(key);
  }
  catch (const std::out_of_range &e)
  {
    spdlog::error("Key not found: N: {} C: {} M: {} P: {} Q: {} S: {} R: {}",
                  key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }
  int core_id = -1; // starts from 0
  for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++)
  {
    for (uint32_t M = 0; M < mapping.tile_out_loop.M; M++)
    {
      for (uint32_t C = 0; C < mapping.tile_out_loop.C; C++)
      {
        if (C == 0)
        {
          core_id = (core_id + 1) % _config.num_cores;
        }
        std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = "Gemm",
            .layer_id = _id,
            .batch = N,
            .Q = 1,
            .P = 1,
            .M = M,
            .C = C,
            .S = 1,
            .R = 1,
            .accum = C != 0,
            .core_id = core_id});
        _tiles.push_back(std::move(tile));
        initialize_instructions(_tiles.back().get(), mapping);
        if (!_tiles.back().get()->instructions.size())
          _tiles.pop_back();
      }
    }
  }
  float total_flops = key.M / ((float)1e3) * key.N / ((float)1e3) * key.C / ((float)1e3) * 2;
  float bias_flops = key.M / ((float)1e3) * key.N / ((float)1e3) / ((float)1e3);
  if (has_bias)
  {
    total_flops += bias_flops;
  }
  spdlog::info("[GemmWs] Keys K = {}, N = {}, M = {}", key.C, key.N, key.M);
  float total_memory = (key.M * key.C + key.N * key.C + key.N * key.M) * _config.precision / ((float)1e9);
  float bias_memory = key.M * _config.precision / ((float)1e9);
  if (has_bias)
  {
    total_memory += bias_memory;
  }
  spdlog::info("[GemmWS]: total {} GFLOPs, {} GB", total_flops, total_memory);
  float theoretical_compute_time = total_flops / _config.max_systolic_flops(target_core);
  float theoretical_mem_time = total_memory / _config.max_dram_bandwidth();
  float theoretical_time = std::max(theoretical_compute_time, theoretical_mem_time);
  spdlog::info("[GemmWS]: Theoretical time(ms): {} Compute time: {} Memory time: {}",
               theoretical_time * 1e3, theoretical_compute_time * 1e3, theoretical_mem_time * 1e3);
}

void GemmWS::initialize_instructions(Tile *tile, Mapping mapping)
{
  int tout_m_offset = tile->M * mapping.tile_in_loop.M;
  int tout_c_offset = tile->C * mapping.tile_in_loop.C;
  int tout_n_offset = tile->batch * mapping.tile_in_loop.N;
  int elems_per_access = _config.dram_req_size / _config.precision;

  addr_type act_sp_base_addr = SPAD_BASE;
  addr_type weight_sp_base_addr = SPAD_BASE + mapping.tile_in_loop.N *
                                                  mapping.tile_in_loop.C *
                                                  _config.precision;

  addr_type first_addr, second_addr, third_addr, output_addr;
  first_addr = get_operand_addr(_INPUT_OPERAND);
  second_addr = get_operand_addr(_INPUT_OPERAND + 1);
  third_addr = get_operand_addr(_INPUT_OPERAND + 2);
  output_addr = get_operand_addr(_OUTPUT_OPERAND);

  int loop_size = _config.core_config[target_core].core_height;
  int cloop_size = mapping.tile_in_loop.C;
  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size)
  {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if (m_loop <= 0)
      break;
    /* MOVIN BIAS */
    if (!tile->accum && has_bias)
    {
      std::vector<addr_type> bias_addrs;
      for (int iter_m = 0; iter_m < m_loop; iter_m += elems_per_access)
      {
        int M = M_offset + iter_m;
        if (M >= mapping.total_loop.M)
          continue;
        bias_addrs.push_back(third_addr + _config.align_address(M * _config.precision));
      }
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = ACCUM_SPAD_BASE + Ms * _config.precision,
          .size = (uint32_t)bias_addrs.size(),
          .src_addrs = std::vector<addr_type>(bias_addrs.begin(), bias_addrs.end()),
          .operand_id = _INPUT_OPERAND + 2}));
    }
    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += cloop_size)
    {
      int C_offset = tout_c_offset + Cs;
      int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                          ? mapping.total_loop.C - C_offset
                          : cloop_size;
      /* MOVIN Weights */
      addr_type weight_sp_addr =
          weight_sp_base_addr +
          (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
      std::set<addr_type> weight_set;
      for (int iter_m = 0; iter_m < m_loop; iter_m += 1)
      {
        for (int iter_c = 0; iter_c < c_in_loop; iter_c += elems_per_access)
        {
          int C = C_offset + iter_c;
          int M = M_offset + iter_m;
          std::vector<uint32_t> weight_shape_2d;
          std::vector<uint32_t> index;
          weight_shape_2d.resize(2);
          index.resize(2);
          weight_shape_2d[1] = _weight_shape[Cdim_w];
          weight_shape_2d[0] = _weight_shape[Mdim];
          index[1] = C;
          index[0] = M;
          weight_set.insert(
              second_addr + make_address(index, weight_shape_2d));
        }
      }
      std::vector<addr_type> weight_addrs =
          make_graphbit_weight_addrs(weight_set, _config);
      Instruction weight_inst = Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = weight_sp_addr,
          .size = (uint32_t)weight_addrs.size(),
          .src_addrs = weight_addrs,
          .operand_id = _INPUT_OPERAND + 1,
          .tile_m = mapping.tile_in_loop.M,
          .tile_k = mapping.tile_in_loop.C,
          .graphbit_original_weight_size = (uint32_t)weight_set.size()};
      annotate_graphbit(weight_inst, static_cast<uint32_t>(mapping.tile_in_loop.C), _config);
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          weight_inst}));

      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size)
      {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                         ? mapping.total_loop.N - N_offset
                         : loop_size;
        if (n_loop <= 0)
          break;
        addr_type act_sp_addr =
            act_sp_base_addr +
            (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;

        /* MOVIN Activation */
        if (Ms == 0)
        {
          std::set<addr_type> input_set;
          for (int iter_n = 0; iter_n < n_loop; iter_n++)
          {
            for (int iter_c = 0; iter_c < c_in_loop; iter_c += elems_per_access)
            {
              uint32_t N = N_offset + iter_n;
              uint32_t C = C_offset + iter_c;
              std::vector<uint32_t> index;
              if (_input_shape.size() == 3)
                index = {N / _input_shape.at(1), N % _input_shape.at(1), C};

              else
                index = {N, C};
              input_set.insert(
                  first_addr + make_address(index, _input_shape));
            }
          }
          uint32_t graphbit_tile_k = static_cast<uint32_t>(
              std::min(static_cast<uint32_t>(c_in_loop),
                       _config.core_config[target_core].core_height));
          Instruction inst = Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = act_sp_addr,
              .operand_id = _INPUT_OPERAND,
              .tile_k = graphbit_tile_k,
              .tile_n = static_cast<unsigned int>(n_loop),
              .graphbit_original_size = (uint32_t)input_set.size()};
          annotate_graphbit(inst, graphbit_tile_k, _config);
          std::vector<addr_type> input_addrs =
              make_graphbit_src_addrs(input_set, inst.graphbit_fetch_depth, _config);
          inst.size = (uint32_t)input_addrs.size();
          inst.src_addrs = input_addrs;
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              inst}));
        }
      }
    }
  }
  /* Compute */
  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size)
  {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if (m_loop <= 0)
      break;
    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += cloop_size)
    {
      int C_offset = tout_c_offset + Cs;
      int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                          ? mapping.total_loop.C - C_offset
                          : cloop_size;
      addr_type weight_sp_addr =
          weight_sp_base_addr +
          (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size)
      {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                         ? mapping.total_loop.N - N_offset
                         : loop_size;
        if (n_loop <= 0)
          break;
        addr_type act_sp_addr =
            act_sp_base_addr +
            (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        for (int c_iter = 0; c_iter < c_in_loop; c_iter += _config.core_config[target_core].core_height)
        {
          int c_iter_size = c_in_loop - c_iter > _config.core_config[target_core].core_height ? _config.core_config[target_core].core_height : c_in_loop - c_iter;
          Instruction inst = Instruction{
              .opcode = Opcode::GEMM_PRELOAD,
              .dest_addr = out_sp_addr,
              .size = (uint32_t)n_loop,
              .compute_size = (uint32_t)n_loop,
              .src_addrs =
                  std::vector<addr_type>{act_sp_addr, weight_sp_addr},
              .tile_m = static_cast<unsigned int>(m_loop),
              .tile_k = static_cast<unsigned int>(c_iter_size),
              .tile_n = static_cast<unsigned int>(n_loop)};
          annotate_graphbit(inst, static_cast<uint32_t>(c_iter_size), _config);
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              inst}));
        }
      }
    }
  }

  /* MOVOUT */
  if (tout_c_offset + mapping.tile_in_loop.C >= mapping.total_loop.C)
  {
    for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size)
    {
      int M_offset = tout_m_offset + Ms;
      int m_loop = M_offset + loop_size > mapping.total_loop.M
                       ? mapping.total_loop.M - M_offset
                       : loop_size;
      if (m_loop <= 0)
        break;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size)
      {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                         ? mapping.total_loop.N - N_offset
                         : loop_size;
        if (n_loop <= 0)
          break;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        std::set<addr_type> output_set;
        for (int iter_n = 0; iter_n < n_loop; iter_n++)
        {
          for (int iter_m = 0; iter_m < m_loop; iter_m += elems_per_access)
          {
            uint32_t N = N_offset + iter_n;
            uint32_t M = M_offset + iter_m;
            std::vector<uint32_t> index;
            if (_output_shape.size() == 3)
              index = {N / _output_shape.at(1), N % _output_shape.at(1), M};
            else
              index = {N, M};
            output_set.insert(output_addr + make_address(index, _output_shape));
          }
        }
        /*MOVOUT result at the last loop*/
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = out_sp_addr,
            .size = (uint32_t)output_set.size(),
            .src_addrs = std::vector<addr_type>(output_set.begin(), output_set.end()),
            .operand_id = _OUTPUT_OPERAND}));
      }
    }
  }
}
