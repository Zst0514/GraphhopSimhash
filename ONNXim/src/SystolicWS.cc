#include "SystolicWS.h"

#include <algorithm>
#include <cmath>

SystolicWS::SystolicWS(uint32_t id, SimulationConfig config)
    : Core(id, config) {}

void SystolicWS::cycle() {
    /*
  Compute unit
  */
  finish_compute_pipeline();
  /* Checking Vector compute pipeline */
  finish_vector_pipeline();
  /* LD in struction queue */
  handle_ld_inst_queue();
  /* EX instruction queue */
  if (!_ex_inst_queue.empty() && can_issue_compute(_ex_inst_queue.front())) { // execution dependecy check
    std::unique_ptr<Instruction> front = std::move(_ex_inst_queue.front());
    if (front->dest_addr >= ACCUM_SPAD_BASE) {
      if (_acc_spad.check_allocated(front->dest_addr, front->accum_spad_id)) {
        _acc_spad.count_up(front->dest_addr, front->accum_spad_id);
      } else {
        int ret = _acc_spad.prefetch(front->dest_addr, front->accum_spad_id, front->size, front->zero_init? front->size : 1);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _acc_spad.check_allocated(front->dest_addr, front->accum_spad_id), _acc_spad.check_remain(front->size, front->accum_spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
          _acc_spad.print_all(front->accum_spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    } else {
      if (_spad.check_allocated(front->dest_addr, front->spad_id)) {
        _spad.count_up(front->dest_addr, front->spad_id);
      } else {
        int ret = _spad.prefetch(front->dest_addr, front->spad_id, front->size, front->zero_init? front->size : 1);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _spad.check_allocated(front->dest_addr, front->spad_id), _spad.check_remain(front->size, front->spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
          _spad.print_all(front->spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    }
    if (front->opcode == Opcode::GEMM || front->opcode == Opcode::GEMM_PRELOAD) {
      if (!_compute_pipeline.empty()) {
        /* Preload can be hided. For Graph-Bit bit-serial execution, the
         * array can accept the next GEMM after the actually executed bit-plane
         * depth, not after the original full-depth compute_size. */
        cycle_type offset = get_inst_issue_spacing(_compute_pipeline.back());
        offset = MAX(offset, 4);
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          // State mul-pre
          offset = MAX(offset, _config.core_config[_id].core_height);
          _stat_systolic_preload_issue_count++;
        }
        if (_compute_pipeline.back()->start_cycle+offset < _core_cycle) {
          front->start_cycle = _core_cycle;
          _stat_systolic_bubble_cycle += (_core_cycle - _compute_pipeline.back()->start_cycle+offset);
        } else
          front->start_cycle = _compute_pipeline.back()->start_cycle+offset;
      } else {
        front->start_cycle = _core_cycle;
        /* Preload weight to systolic array*/
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          /* Weight preload  from buffer latecny + WEight preload latency */
          front->start_cycle += _config.core_config[_id].core_height + _config.core_config[_id].core_height - 1;
          _stat_systolic_preload_issue_count++;
        }
      }
      front->finish_cycle = front->start_cycle + get_inst_compute_cycles(front);
      if (front->graphbit_enabled) {
        uint32_t full_depth = std::max(1u, front->graphbit_full_depth);
        uint32_t effective_depth =
            std::min(front->graphbit_effective_depth, full_depth);
        _stat_graphbit_inst_count++;
        _stat_graphbit_effective_bitplanes += effective_depth;
        _stat_graphbit_saved_bitplanes += (full_depth - effective_depth);
        if (front->graphbit_remaining_bound <=
            _config.graphbit_bound_tolerance) {
          _stat_graphbit_bound_stop_count++;
        }
      }
      _compute_pipeline.push(std::move(front));
      _stat_systolic_inst_issue_count++;
    } else {  // vector unit compute
      front->start_cycle = _core_cycle;
      front->finish_cycle =
          front->start_cycle +
          get_vector_compute_cycles(front);  // Setting IC as 1 (Might need to modify)
      _vector_pipeline.push(std::move(front));
    }
    _ex_inst_queue.pop();
  }

  /* ST in struction queue */
  handle_st_inst_queue();

  // xxx will it work well on double buffered code? no.
  bool is_idle = _compute_pipeline.empty() && _vector_pipeline.empty();
  bool is_running = running();
  bool is_compute_busy = false;
  bool is_vector_busy = false;

  if (!_compute_pipeline.empty() && _compute_pipeline.front()->start_cycle <= _core_cycle)
    is_compute_busy = true;
  if (!_vector_pipeline.empty() && _vector_pipeline.front()->start_cycle <= _core_cycle)
    is_vector_busy = true;

  if (is_compute_busy)
    _stat_systolic_active_cycle++;
  if (is_vector_busy)
    _stat_vec_compute_cycle++;

  if (is_compute_busy || is_vector_busy)
    _stat_compute_cycle++;

  if (_request_queue.empty())
    _stat_memory_idle_cycle++;

  if (!is_running)
    _stat_idle_cycle++;
  Core::cycle();
}

bool SystolicWS::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  if(Core::can_issue_compute(inst) == false)
    return false;
  if (inst->opcode == Opcode::GEMM || inst->opcode == Opcode::GEMM_PRELOAD) {
    if (_compute_pipeline.size() >= _config.core_config[_id].core_height) {
      return false;
    }
  } else {
    if(!_vector_pipeline.empty()) {
      return false;
    }
  }
  return true;
}

cycle_type SystolicWS::get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) {
  cycle_type raw_cycles =
      _config.core_config[_id].core_height + _config.core_config[_id].core_width -
      2 + MAX(inst->compute_size, 4);
  if (!inst->graphbit_enabled) {
    return raw_cycles;
  }
  uint32_t full_depth = std::max(1u, inst->graphbit_full_depth);
  uint32_t effective_depth =
      std::max(1u, std::min(inst->graphbit_effective_depth, full_depth));
  cycle_type scaled_cycles = static_cast<cycle_type>(
      std::ceil(static_cast<double>(raw_cycles) *
                static_cast<double>(effective_depth) /
                static_cast<double>(full_depth)));
  return std::max<cycle_type>(1, scaled_cycles);
}

cycle_type SystolicWS::get_inst_issue_spacing(std::unique_ptr<Instruction>& inst) {
  cycle_type raw_spacing = inst->compute_size;
  if (!inst->graphbit_enabled) {
    return raw_spacing;
  }
  uint32_t full_depth = std::max(1u, inst->graphbit_full_depth);
  uint32_t effective_depth =
      std::max(1u, std::min(inst->graphbit_effective_depth, full_depth));
  cycle_type scaled_spacing = static_cast<cycle_type>(
      std::ceil(static_cast<double>(raw_spacing) *
                static_cast<double>(effective_depth) /
                static_cast<double>(full_depth)));
  return std::max<cycle_type>(1, scaled_spacing);
}

cycle_type SystolicWS::get_vector_compute_cycles(std::unique_ptr<Instruction>& inst) {
  cycle_type vec_op_iter = calculate_vector_op_iterations(inst->compute_size);
  cycle_type add_tree_iter = calculate_add_tree_iterations(inst->compute_size);
  cycle_type add_tree, scalar_ops, vector_ops;
  switch (inst->opcode) {
    case Opcode::LAYERNORM:
      add_tree = 2 * add_tree_iter * _config.core_config[_id].add_tree_latency;
      scalar_ops = 2 * _config.core_config[_id].scalar_mul_latency + _config.core_config[_id].scalar_sqrt_latency;
      // 1 addition, 1 subtraction, 1 division, 2 multiplication.
      vector_ops = vec_op_iter * (2 * _config.core_config[_id].add_latency + 3 * _config.core_config[_id].mul_latency) * inst->tile_m;
      return add_tree + scalar_ops + vector_ops;
    case Opcode::SOFTMAX:
      // 1 add tree, 1 compare tree
      add_tree = 2 * add_tree_iter * _config.core_config[_id].add_tree_latency * inst->tile_m;
      vector_ops =
        vec_op_iter * (_config.core_config[_id].add_latency + _config.core_config[_id].exp_latency + _config.core_config[_id].mul_latency);
      return add_tree + vector_ops;
    case Opcode::ADD:
      return vec_op_iter * _config.core_config[_id].add_latency;
    case Opcode::MUL:
      return vec_op_iter * _config.core_config[_id].mul_latency;
    case Opcode::MAC:
      return vec_op_iter * _config.core_config[_id].mac_latency;
    case Opcode::SWISH: //TODO: Implement SWISH
    case Opcode::GELU:
      return vec_op_iter * _config.core_config[_id].gelu_latency;
    case Opcode::COMP:
      return vec_op_iter * 1;
    case Opcode::ADDTREE:
      return add_tree_iter * _config.core_config[_id].add_tree_latency * inst->tile_m;
    case Opcode::DIV:
      return vec_op_iter * _config.core_config[_id].div_latency;
    case Opcode::EXP:
      return vec_op_iter * _config.core_config[_id].exp_latency;
    
  }
  spdlog::info("not configured operation. {}", inst->id);
  // assert(0);
  return 0;
}

void SystolicWS::print_stats() {
  Core::print_stats();
  spdlog::info("Core [{}] : Systolic Inst Issue Count : {}", _id,
               _stat_systolic_inst_issue_count);
  spdlog::info("Core [{}] : Systolic PRELOAD Issue Count : {}", _id,
               _stat_systolic_preload_issue_count);
  if (_stat_graphbit_inst_count > 0) {
    double avg_depth =
        static_cast<double>(_stat_graphbit_effective_bitplanes) /
        static_cast<double>(_stat_graphbit_inst_count);
    double avg_saved =
        static_cast<double>(_stat_graphbit_saved_bitplanes) /
        static_cast<double>(_stat_graphbit_inst_count);
    spdlog::info(
        "Core [{}] : GraphBit Inst {} BoundStops {} AvgDepth {:.2f} "
        "AvgSavedBitplanes {:.2f}",
        _id, _stat_graphbit_inst_count, _stat_graphbit_bound_stop_count,
        avg_depth, avg_saved);
  }
}
