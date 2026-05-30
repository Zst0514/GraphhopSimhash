#include "Core.h"

class SystolicWS : public Core {
 public:
  SystolicWS(uint32_t id, SimulationConfig config);
  virtual void cycle() override;
  virtual void print_stats() override;

 protected:
  virtual bool can_issue_compute(std::unique_ptr<Instruction>& inst) override;
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) override;
  cycle_type get_inst_issue_spacing(std::unique_ptr<Instruction>& inst);
  uint32_t _stat_systolic_inst_issue_count = 0;
  uint32_t _stat_systolic_preload_issue_count = 0;
  uint64_t _stat_graphbit_inst_count = 0;
  uint64_t _stat_graphbit_bound_stop_count = 0;
  uint64_t _stat_graphbit_effective_bitplanes = 0;
  uint64_t _stat_graphbit_saved_bitplanes = 0;
  uint64_t _stat_graphbit_fetch_bitplanes = 0;
  uint64_t _stat_graphbit_issue_bitplanes = 0;
  uint64_t _stat_graphbit_weight_bitplanes = 0;
  uint64_t _stat_graphbit_psum_bitplanes = 0;
  double _stat_graphbit_effective_compute_cycles = 0;
  double _stat_graphbit_raw_compute_cycles = 0;
  cycle_type get_inst_raw_compute_cycles(std::unique_ptr<Instruction>& inst);
  cycle_type get_vector_compute_cycles(std::unique_ptr<Instruction>& inst);
};
