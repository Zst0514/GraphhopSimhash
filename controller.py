from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn.functional as F

from .features import _compute_neighbor_mean
from .projections import build_hash_random_matrix
from .scoring import (
    QuantExecutionPolicy,
    QuantPolicyConfig,
    ReuseRiskGate,
    RiskGateConfig,
    build_node_risk_scores,
    summarize_scores,
)

class HeatPlusPlus_NDP_Controller:
    def __init__(self, input_dim, sketch_bits=64, device="cuda", hamming_radius=2, random_seed=42):
        self.sketch_bits = sketch_bits
        self.device = device
        self.hamming_radius = hamming_radius
        self.random_seed = int(random_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)
        self.R = torch.randn(input_dim, sketch_bits, generator=generator).to(device)
        self.memo_table = {}
        self.seen_nodes = set()
        self.mih_enabled = sketch_bits >= 32 and hamming_radius > 0
        if self.mih_enabled:
            self.num_chunks = 4
            self.chunk_len = (sketch_bits + 3) // 4
            self.mih_tables = [{} for _ in range(self.num_chunks)]

        print(
            f"[Heat++] Bits={sketch_bits}, Radius={self.hamming_radius}, "
            f"InitSeed={self.random_seed}, MIH Accelerated={self.mih_enabled}"
        )

        self.use_segments = True
        if self.use_segments:
            self.seg_len = 4
            self.num_segs = sketch_bits // 4
            self.seg_vote_thresh = 2 if sketch_bits >= 64 else 1
            self.seg_tables = [{} for _ in range(self.num_segs)]
            print(
                f"[Heat++] Activated Segmented Filtering: "
                f"{self.num_segs}x{self.seg_len}-bit, VoteThresh={self.seg_vote_thresh}"
            )

        self.stats = {
            "total_queries": 0,
            "reuse": 0,
            "fuzzy": 0,
            "computed": 0,
        }
class PaperHashReuseController(HeatPlusPlus_NDP_Controller):
    def __init__(
        self,
        input_dim,
        sketch_bits,
        device,
        hamming_radius,
        full_verify_features,
        full_hash_features,
        edge_index,
        full_hash_feature_routes=None,
        hash_route_matrices=None,
        hash_route_bits=None,
        hash_route_names=None,
        route_base_indices=None,
        route_base_names=None,
        route_score_weights=None,
        route_accept_tau_offsets=None,
        route_min_accept_votes=None,
        route_min_support_hits=None,
        min_base_route_hits=1,
        max_candidates_per_route=64,
        max_total_candidates=128,
        max_structure_checks=None,
        coarse_union_bits_max=None,
        max_cache_size=None,
        second_stage_tau=0.90,
        memo_k=3,
        vote_top_m=4,
        vote_relax_margin=0.05,
        structure_neighbor_tau=0.45,
        structure_degree_ratio_max=4.0,
        structure_homophily_gap_max=None,
        structure_check_mode="sketch",
        topology_sketch_bits=32,
        topology_sketch_radius=10,
        enable_topology_sketch_guard=False,
        topology_degree_bucket_gap=None,
        topology_homophily_bins=8,
        topology_homophily_bucket_gap=3,
        topology_sketch_seed=424242,
        exact_guard_low_bits=16,
        exact_guard_min_bucket_size=2,
        exact_guard_large_bucket_size=4,
        exact_guard_min_margin=0.03,
        exact_guard_cosine_bonus=0.02,
        hamming_only_acceptor=False,
        disable_structure_check=False,
        score_gate_enabled=True,
        score_reuse_threshold=120,
        score_hub_threshold=12,
        score_rare_threshold=10,
        score_protect_hub_exact=False,
        score_protect_hub_fuzzy=True,
        score_forbid_rare_fuzzy=True,
        score_support_discount=True,
        score_rarity_bits=16,
        score_rarity_seed=98765,
        score_propagation_weight=3,
        score_graph_context_weight=2,
        score_low_unique_weight=2,
        quant_policy_enabled=False,
        quant_int4_threshold=90,
        quant_int8_threshold=45,
        quant_int4_error=3,
        quant_int8_error=1,
        quant_int4_bits=4,
        quant_int8_bits=8,
        hash_init_seed=42,
    ):
        self.max_cache_size = max_cache_size
        self.second_stage_tau = second_stage_tau
        self.memo_k = memo_k
        self.vote_top_m = vote_top_m
        self.vote_relax_margin = vote_relax_margin
        self.max_candidates_per_route = int(max_candidates_per_route)
        self.max_total_candidates = int(max_total_candidates)
        self.max_structure_checks = None if max_structure_checks is None else int(max_structure_checks)
        self.coarse_union_bits_max = None if coarse_union_bits_max is None else int(coarse_union_bits_max)
        self.structure_check_mode = str(structure_check_mode)
        self.topology_sketch_bits = int(topology_sketch_bits)
        self.topology_sketch_radius = int(topology_sketch_radius)
        self.enable_topology_sketch_guard = bool(enable_topology_sketch_guard)
        self.topology_degree_bucket_gap = None if topology_degree_bucket_gap is None else int(topology_degree_bucket_gap)
        self.topology_homophily_bins = int(topology_homophily_bins)
        self.topology_homophily_bucket_gap = int(topology_homophily_bucket_gap)
        self.topology_sketch_seed = int(topology_sketch_seed)
        self.exact_guard_low_bits = exact_guard_low_bits
        self.exact_guard_min_bucket_size = exact_guard_min_bucket_size
        self.exact_guard_large_bucket_size = exact_guard_large_bucket_size
        self.exact_guard_min_margin = exact_guard_min_margin
        self.exact_guard_cosine_bonus = exact_guard_cosine_bonus
        self.hamming_only_acceptor = bool(hamming_only_acceptor)
        self.disable_structure_check = bool(disable_structure_check)
        self.score_gate_config = RiskGateConfig(
            enabled=bool(score_gate_enabled),
            reuse_threshold=int(score_reuse_threshold),
            hub_threshold=int(score_hub_threshold),
            rare_threshold=int(score_rare_threshold),
            protect_hub_exact=bool(score_protect_hub_exact),
            protect_hub_fuzzy=bool(score_protect_hub_fuzzy),
            forbid_rare_fuzzy=bool(score_forbid_rare_fuzzy),
            support_discount=bool(score_support_discount),
        )
        self.score_rarity_bits = int(score_rarity_bits)
        self.score_rarity_seed = int(score_rarity_seed)
        self.score_propagation_weight = int(score_propagation_weight)
        self.score_graph_context_weight = int(score_graph_context_weight)
        self.score_low_unique_weight = int(score_low_unique_weight)
        self.quant_policy_config = QuantPolicyConfig(
            enabled=bool(quant_policy_enabled),
            int4_threshold=int(quant_int4_threshold),
            int8_threshold=int(quant_int8_threshold),
            int4_error=int(quant_int4_error),
            int8_error=int(quant_int8_error),
        )
        self.quant_int4_bits = int(quant_int4_bits)
        self.quant_int8_bits = int(quant_int8_bits)
        self.hash_init_seed = int(hash_init_seed)
        self.node_risk_scores = None
        self.risk_gate = None
        self.quant_policy = None
        self.score_summary = None
        self._time_counter = 0

        super().__init__(input_dim, sketch_bits, device, hamming_radius, random_seed=self.hash_init_seed)

        self.memo_table = OrderedDict()
        self.hash_population = {}
        self.full_verify_features = full_verify_features
        self.full_hash_features = full_hash_features
        self.edge_index = edge_index
        self.full_hash_feature_routes = list(full_hash_feature_routes or [full_hash_features])
        if hash_route_matrices is None:
            hash_route_matrices = [self.R] * len(self.full_hash_feature_routes)
        if len(hash_route_matrices) != len(self.full_hash_feature_routes):
            raise ValueError("hash_route_matrices must match full_hash_feature_routes length")
        self.hash_route_matrices = [matrix.to(self.device) for matrix in hash_route_matrices]
        if hash_route_bits is None:
            hash_route_bits = [int(matrix.size(1)) for matrix in self.hash_route_matrices]
        if len(hash_route_bits) != len(self.full_hash_feature_routes):
            raise ValueError("hash_route_bits must match full_hash_feature_routes length")
        self.hash_route_bits = [int(bits) for bits in hash_route_bits]

        if hash_route_names is None:
            hash_route_names = [f"route_{idx}" for idx in range(len(self.full_hash_feature_routes))]
        if len(hash_route_names) != len(self.full_hash_feature_routes):
            raise ValueError("hash_route_names must match full_hash_feature_routes length")
        self.hash_route_names = list(hash_route_names)

        if route_base_indices is None:
            route_base_indices = list(range(len(self.hash_route_names)))
        if len(route_base_indices) != len(self.hash_route_names):
            raise ValueError("route_base_indices must match full_hash_feature_routes length")
        self.route_base_indices = [int(idx) for idx in route_base_indices]

        max_base_idx = max(self.route_base_indices) if self.route_base_indices else -1
        if route_base_names is None:
            route_base_names = [self.hash_route_names[idx] for idx in range(max_base_idx + 1)]
        if len(route_base_names) <= max_base_idx:
            raise ValueError("route_base_names must cover every base route index")
        self.route_base_names = list(route_base_names)

        if route_score_weights is None:
            route_score_weights = [1.0] * len(self.hash_route_names)
        if len(route_score_weights) != len(self.hash_route_names):
            raise ValueError("route_score_weights must match full_hash_feature_routes length")
        self.route_score_weights = [float(weight) for weight in route_score_weights]

        if route_accept_tau_offsets is None:
            route_accept_tau_offsets = [0.0] * len(self.hash_route_names)
        if len(route_accept_tau_offsets) != len(self.hash_route_names):
            raise ValueError("route_accept_tau_offsets must match full_hash_feature_routes length")
        self.route_accept_tau_offsets = [float(offset) for offset in route_accept_tau_offsets]

        if route_min_accept_votes is None:
            route_min_accept_votes = [1] * len(self.hash_route_names)
        if len(route_min_accept_votes) != len(self.hash_route_names):
            raise ValueError("route_min_accept_votes must match full_hash_feature_routes length")
        self.route_min_accept_votes = [int(votes) for votes in route_min_accept_votes]

        if route_min_support_hits is None:
            route_min_support_hits = [1] * len(self.route_base_names)
        if len(route_min_support_hits) != len(self.route_base_names):
            raise ValueError("route_min_support_hits must match route_base_names length")
        self.route_min_support_hits = [int(hits) for hits in route_min_support_hits]
        self.min_base_route_hits = max(1, int(min_base_route_hits))

        self.hash_routes = [
            self._build_hash_route_state(route_name, route_bits)
            for route_name, route_bits in zip(self.hash_route_names, self.hash_route_bits)
        ]
        self.memo_table = self.hash_routes[0]["memo_table"]
        self.hash_population = self.hash_routes[0]["hash_population"]
        self.node_policies = None
        self.node_total_degree = None
        self.node_context_signature = None
        self.node_avg_homophily = None
        self.node_topology_sketch = None
        self.node_degree_bucket = None
        self.node_homophily_bucket = None
        self.structure_neighbor_tau = structure_neighbor_tau
        self.structure_degree_ratio_max = structure_degree_ratio_max
        if structure_homophily_gap_max is None:
            self.structure_homophily_gap_max = max(1.0, float(self.hash_route_bits[0]) * 0.20)
        else:
            self.structure_homophily_gap_max = structure_homophily_gap_max

        self._precompute_structure_context()
        self._compute_homophily_policies()
        self._precompute_topology_sketch_guard()
        self._precompute_risk_scores()

    def _build_hash_route_state(self, route_name, route_bits):
        route_bits = int(route_bits)
        seg_len = 4
        num_segs = route_bits // seg_len
        use_segments = num_segs > 0
        mih_enabled = route_bits >= 32 and self.hamming_radius > 0
        chunk_len = (route_bits + 3) // 4 if mih_enabled else None
        num_chunks = 4 if mih_enabled else None
        route_state = {
            "name": route_name,
            "route_bits": route_bits,
            "memo_table": OrderedDict(),
            "hash_population": {},
            "mih_enabled": mih_enabled,
            "num_chunks": num_chunks,
            "chunk_len": chunk_len,
            "use_segments": use_segments,
            "seg_len": seg_len,
            "num_segs": num_segs,
        }
        if mih_enabled:
            route_state["mih_tables"] = [{} for _ in range(num_chunks)]
        else:
            route_state["mih_tables"] = None
        if use_segments:
            route_state["seg_tables"] = [{} for _ in range(num_segs)]
        else:
            route_state["seg_tables"] = None
        return route_state

    def _get_route_state(self, route_idx):
        return self.hash_routes[route_idx]

    def _normalize_hash_feature_routes(self, hash_features):
        if isinstance(hash_features, (list, tuple)):
            routes = list(hash_features)
        else:
            routes = [hash_features]
        if len(routes) != len(self.hash_routes):
            raise ValueError(
                f"Expected {len(self.hash_routes)} hash feature routes, got {len(routes)}"
            )
        return routes

    def _compute_fingerprint_with_matrix(self, x, matrix):
        proj = torch.matmul(x, matrix)
        bits = (proj > 0).cpu().numpy().astype(int)

        vals = []
        for row in bits:
            val = 0
            for bit in row:
                val = (val << 1) | int(bit)
            vals.append(val)
        return vals

    def _compute_route_fingerprint(self, route_features, route_idx):
        return self._compute_fingerprint_with_matrix(route_features, self.hash_route_matrices[route_idx])

    def _get_route_chunks(self, h, route_idx):
        route_state = self._get_route_state(route_idx)
        chunk_len = route_state["chunk_len"]
        num_chunks = route_state["num_chunks"]
        chunks = []
        mask = (1 << chunk_len) - 1
        temp_h = h
        for _ in range(num_chunks):
            chunks.append(temp_h & mask)
            temp_h >>= chunk_len
        return chunks

    def _get_route_segments(self, h, route_idx):
        route_state = self._get_route_state(route_idx)
        seg_len = route_state["seg_len"]
        num_segs = route_state["num_segs"]
        segs = []
        mask = (1 << seg_len) - 1
        temp_h = h
        for _ in range(num_segs):
            segs.append(temp_h & mask)
            temp_h >>= seg_len
        return segs

    def _add_to_route_mih(self, route_idx, h):
        route_state = self._get_route_state(route_idx)
        if not route_state["mih_enabled"]:
            return
        chunks = self._get_route_chunks(h, route_idx)
        for chunk_idx, chunk in enumerate(chunks):
            if chunk not in route_state["mih_tables"][chunk_idx]:
                route_state["mih_tables"][chunk_idx][chunk] = []
            route_state["mih_tables"][chunk_idx][chunk].append(h)

    def _add_to_route_segments(self, route_idx, h):
        route_state = self._get_route_state(route_idx)
        if not route_state["use_segments"]:
            return
        segs = self._get_route_segments(h, route_idx)
        for seg_idx, seg in enumerate(segs):
            if seg not in route_state["seg_tables"][seg_idx]:
                route_state["seg_tables"][seg_idx][seg] = []
            route_state["seg_tables"][seg_idx][seg].append(h)

    def _touch_route_entry(self, route_idx, cand_hash, best_entry):
        route_state = self._get_route_state(route_idx)
        entries = route_state["memo_table"].get(cand_hash, [])
        try:
            route_state["memo_table"].move_to_end(cand_hash)
        except Exception:
            pass
        best_entry["timestamp"] = self._time_counter
        self._time_counter += 1
        try:
            entries.remove(best_entry)
            entries.insert(0, best_entry)
        except Exception:
            pass

    def _reset_route_caches(self):
        self.hash_routes = [
            self._build_hash_route_state(route_name, route_bits)
            for route_name, route_bits in zip(self.hash_route_names, self.hash_route_bits)
        ]
        self.memo_table = self.hash_routes[0]["memo_table"]
        self.hash_population = self.hash_routes[0]["hash_population"]
        self._time_counter = 0

    def _cache_computed_entry(self, query_hashes, entry):
        for route_idx, route_hash in enumerate(query_hashes):
            route_state = self._get_route_state(route_idx)
            route_state["hash_population"][route_hash] = route_state["hash_population"].get(route_hash, 0) + 1

            if route_hash not in route_state["memo_table"]:
                route_state["memo_table"][route_hash] = [entry]
                route_state["memo_table"].move_to_end(route_hash)
                if self.max_cache_size is not None and len(route_state["memo_table"]) > self.max_cache_size:
                    route_state["memo_table"].popitem(last=False)
            else:
                entries = route_state["memo_table"][route_hash]
                entries.insert(0, entry)
                if len(entries) > self.memo_k:
                    entries.pop(-1)
                try:
                    route_state["memo_table"].move_to_end(route_hash)
                except Exception:
                    pass

            self._add_to_route_mih(route_idx, route_hash)
            self._add_to_route_segments(route_idx, route_hash)

    def _precompute_structure_context(self):
        neighbor_mean = _compute_neighbor_mean(self.full_verify_features, self.edge_index)
        row, col = self.edge_index
        sym_row = torch.cat([row, col], dim=0)
        total_degree = torch.zeros(self.full_verify_features.size(0), device=self.device)
        total_degree.index_add_(0, sym_row, torch.ones(sym_row.size(0), device=self.device))
        self.node_total_degree = total_degree
        self.node_context_signature = F.normalize(
            0.5 * self.full_verify_features + 0.5 * neighbor_mean,
            p=2,
            dim=1,
        )

    def _compute_homophily_policies(self):
        num_nodes = self.full_hash_features.size(0)
        self.node_policies = torch.zeros(num_nodes, dtype=torch.int, device=self.device)

        proj = torch.matmul(self.full_hash_features, self.hash_route_matrices[0])
        fingerprints_bits = (proj > 0).float()

        row, col = self.edge_index
        src_bits = fingerprints_bits[row]
        dst_bits = fingerprints_bits[col]
        edge_dists = (src_bits != dst_bits).sum(dim=1).float()

        node_sum_dist = torch.zeros(num_nodes, device=self.device)
        node_degree = torch.zeros(num_nodes, device=self.device)
        node_sum_dist.index_add_(0, row, edge_dists)
        node_degree.index_add_(0, row, torch.ones_like(edge_dists))

        avg_homophily = node_sum_dist / node_degree.clamp(min=1)
        self.node_avg_homophily = avg_homophily

        active_radius = max(0, int(self.hamming_radius))
        self.node_policies[:] = active_radius

        active_count = (self.node_policies == active_radius).sum().item()
        r0 = (self.node_policies == 0).sum().item()
        print(
            f"[Radius] Fixed policy: R={active_radius} {active_count/num_nodes:.1%}, "
            f"R=0 {r0/num_nodes:.1%}"
        )

    def _precompute_topology_sketch_guard(self):
        if self.structure_check_mode != "sketch":
            return

        degree_plus_one = self.node_total_degree + 1.0
        self.node_degree_bucket = torch.floor(torch.log2(degree_plus_one)).to(torch.int16)

        if self.node_avg_homophily is None:
            self.node_homophily_bucket = torch.zeros_like(self.node_degree_bucket)
        else:
            num_bins = max(2, int(self.topology_homophily_bins))
            quantiles = torch.linspace(0.0, 1.0, num_bins + 1, device=self.device)[1:-1]
            edges = torch.quantile(self.node_avg_homophily, quantiles)
            if edges.numel() > 0:
                edges = torch.unique(edges)
            if edges.numel() == 0:
                self.node_homophily_bucket = torch.zeros_like(self.node_degree_bucket)
            else:
                self.node_homophily_bucket = torch.bucketize(self.node_avg_homophily, edges).to(torch.int16)

        if not self.enable_topology_sketch_guard:
            self.node_topology_sketch = None
            return

        sketch_matrix = build_hash_random_matrix(
            self.node_context_signature.size(1),
            self.topology_sketch_bits,
            self.device,
            self.topology_sketch_seed,
        )
        self.node_topology_sketch = self._compute_fingerprint_with_matrix(self.node_context_signature, sketch_matrix)

    def _precompute_risk_scores(self):
        if not self.score_gate_config.enabled and not self.quant_policy_config.enabled:
            print("[RiskGate] disabled")
            self.node_risk_scores = None
            self.risk_gate = None
            self.quant_policy = None
            self.score_summary = None
            return

        self.node_risk_scores = build_node_risk_scores(
            verify_features=self.full_verify_features,
            hash_features=self.full_hash_features,
            edge_index=self.edge_index,
            total_degree=self.node_total_degree,
            context_signature=self.node_context_signature,
            hash_matrix=self.hash_route_matrices[0],
            rarity_bits=self.score_rarity_bits,
            rarity_seed=self.score_rarity_seed,
            propagation_weight=self.score_propagation_weight,
            graph_context_weight=self.score_graph_context_weight,
            low_unique_weight=self.score_low_unique_weight,
        )
        self.risk_gate = (
            ReuseRiskGate(self.node_risk_scores, self.score_gate_config)
            if self.score_gate_config.enabled
            else None
        )
        self.quant_policy = (
            QuantExecutionPolicy(self.node_risk_scores, self.quant_policy_config)
            if self.quant_policy_config.enabled
            else None
        )
        self.score_summary = summarize_scores(self.node_risk_scores)
        gate_status = "enabled" if self.score_gate_config.enabled else "disabled"
        quant_status = "enabled" if self.quant_policy_config.enabled else "disabled"
        print(
            "[RiskScore] "
            f"reuse_gate={gate_status} | quant_policy={quant_status} "
            f"| T_reuse={self.score_gate_config.reuse_threshold} "
            f"| T_hub={self.score_gate_config.hub_threshold} "
            f"| T_rare={self.score_gate_config.rare_threshold}"
        )
        if self.quant_policy_config.enabled:
            print(
                "[QuantPolicy] "
                f"T_int4={self.quant_policy_config.int4_threshold} "
                f"| T_int8={self.quant_policy_config.int8_threshold} "
                f"| E_int4={self.quant_policy_config.int4_error} "
                f"| E_int8={self.quant_policy_config.int8_error} "
                f"| fake_bits={self.quant_int4_bits}/{self.quant_int8_bits}"
            )
        for name, (vmin, vmean, vmax) in self.score_summary.items():
            print(f"[RiskScore] {name}: min={vmin:.1f}, mean={vmean:.1f}, max={vmax:.1f}")

    def _score_gate_allows(self, query_node_id, item):
        if self.risk_gate is None:
            return True, None
        decision = self.risk_gate.evaluate(
            query_node_id,
            item["dist"],
            route_hit_count=item.get("route_hit_count", 1),
        )
        self.stats["score_checked"] += 1
        self.stats["score_risk_sum"] += decision["risk"]
        self.stats["score_sensitivity_sum"] += decision["sensitivity"]
        self.stats["score_count"] += 1
        if not decision["allow"]:
            self.stats["score_reject"] += 1
            reason_key = f"score_reject_{decision['reason']}"
            if reason_key in self.stats:
                self.stats[reason_key] += 1
            return False, decision
        return True, decision

    def _fake_quantize_embedding(self, emb, bits):
        bits = int(bits)
        if bits >= 16:
            return emb
        levels = max(1, (1 << max(1, bits - 1)) - 1)
        max_abs = emb.detach().abs().max().clamp(min=1e-8)
        scale = max_abs / float(levels)
        quantized = torch.round(emb / scale).clamp(-levels, levels) * scale
        return quantized

    def _materialize_computed_embedding(self, node_idx, real_emb):
        if self.quant_policy is None:
            self.stats["full_precision"] += 1
            return real_emb, {
                "action": "full_precision",
                "reason": "quant_disabled",
                "sensitivity": 0,
                "int4_risk": 0,
                "int8_risk": 0,
            }

        decision = self.quant_policy.decide(node_idx)
        self.stats["quant_checked"] += 1
        self.stats["quant_sensitivity_sum"] += decision["sensitivity"]

        action = decision["action"]
        if action == "int4":
            self.stats["quant_int4"] += 1
            self.stats["quant_risk_sum"] += decision["int4_risk"]
            return self._fake_quantize_embedding(real_emb, self.quant_int4_bits), decision
        if action == "int8":
            self.stats["quant_int8"] += 1
            self.stats["quant_risk_sum"] += decision["int8_risk"]
            return self._fake_quantize_embedding(real_emb, self.quant_int8_bits), decision
        if action == "protected":
            self.stats["protected"] += 1
            self.stats["quant_risk_sum"] += decision["int8_risk"]
            return real_emb, decision

        self.stats["full_precision"] += 1
        return real_emb, decision

    def find_fuzzy_candidate_hashes_adaptive(self, h, allowed_r, route_idx=0, max_candidates=64):
        if allowed_r == 0:
            return []

        route_state = self._get_route_state(route_idx)
        ranked_candidates = []
        seen_hashes = set()
        route_bits = int(route_state["route_bits"])

        if route_state["mih_enabled"]:
            chunks = self._get_route_chunks(h, route_idx)
            candidates = set()
            for chunk_idx, chunk in enumerate(chunks):
                if chunk in route_state["mih_tables"][chunk_idx]:
                    candidates.update(route_state["mih_tables"][chunk_idx][chunk])

            for cand in candidates:
                if cand == h or cand in seen_hashes:
                    continue
                if cand not in route_state["memo_table"] or len(route_state["memo_table"][cand]) == 0:
                    continue
                dist = bin(h ^ cand).count("1")
                if dist <= allowed_r:
                    ranked_candidates.append((dist, 0, cand))
                    seen_hashes.add(cand)

            ranked_candidates.sort(key=lambda item: (item[0], item[1]))
            return [cand for _, _, cand in ranked_candidates[:max_candidates]]

        if route_state["use_segments"]:
            segs = self._get_route_segments(h, route_idx)
            candidates = {}
            for seg_idx, seg in enumerate(segs):
                if seg in route_state["seg_tables"][seg_idx]:
                    for cand_h in route_state["seg_tables"][seg_idx][seg]:
                        candidates[cand_h] = candidates.get(cand_h, 0) + 1

            for cand_h, votes in candidates.items():
                if cand_h == h or cand_h in seen_hashes:
                    continue
                if cand_h not in route_state["memo_table"] or len(route_state["memo_table"][cand_h]) == 0:
                    continue
                if votes < self.seg_vote_thresh:
                    continue
                dist = bin(h ^ cand_h).count("1")
                if dist <= allowed_r:
                    ranked_candidates.append((dist, -votes, cand_h))
                    seen_hashes.add(cand_h)

            ranked_candidates.sort(key=lambda item: (item[0], item[1]))
            return [cand for _, _, cand in ranked_candidates[:max_candidates]]

        for radius in range(1, allowed_r + 1):
            for bit_ids in combinations(range(route_bits), radius):
                neighbor = h
                for bit_id in bit_ids:
                    neighbor ^= (1 << bit_id)
                if neighbor == h or neighbor in seen_hashes:
                    continue
                if neighbor in route_state["memo_table"] and len(route_state["memo_table"][neighbor]) > 0:
                    ranked_candidates.append((radius, 0, neighbor))
                    seen_hashes.add(neighbor)
                if len(ranked_candidates) >= max_candidates:
                    break
            if len(ranked_candidates) >= max_candidates:
                break

        ranked_candidates.sort(key=lambda item: (item[0], item[1]))
        return [cand for _, _, cand in ranked_candidates[:max_candidates]]

    def _select_best_entry(self, entries, query_feat):
        best_entry = None
        best_cos = -1.0
        for entry in entries:
            try:
                cos = F.cosine_similarity(entry["cheap_feat"].unsqueeze(0), query_feat.unsqueeze(0)).item()
            except Exception:
                cos = F.cosine_similarity(
                    entry["cheap_feat"].cpu().unsqueeze(0),
                    query_feat.cpu().unsqueeze(0),
                ).item()
            if cos > best_cos:
                best_cos = cos
                best_entry = entry
        return best_entry, best_cos

    def _compute_entry_cosine(self, entry, query_feat):
        try:
            return F.cosine_similarity(entry["cheap_feat"].unsqueeze(0), query_feat.unsqueeze(0)).item()
        except Exception:
            return F.cosine_similarity(
                entry["cheap_feat"].cpu().unsqueeze(0),
                query_feat.cpu().unsqueeze(0),
            ).item()

    def _compute_hamming_match_score(self, route_idx, hamming_dist):
        route_bits = max(1, int(self.hash_route_bits[route_idx]))
        return max(0.0, 1.0 - (float(hamming_dist) / float(route_bits)))

    def _compute_local_structure_metrics(self, query_node_id, entry):
        cand_node_id = int(entry["node_id"])
        query_degree = float(self.node_total_degree[query_node_id].item())
        cand_degree = float(self.node_total_degree[cand_node_id].item())
        if cand_node_id == query_node_id:
            return {
                "neighbor_cos": 1.0,
                "degree_ratio": 1.0,
                "homophily_gap": 0.0,
                "query_degree": query_degree,
                "cand_degree": cand_degree,
            }

        query_sig = self.node_context_signature[query_node_id]
        cand_sig = self.node_context_signature[cand_node_id]
        neighbor_cos = F.cosine_similarity(query_sig.unsqueeze(0), cand_sig.unsqueeze(0)).item()
        degree_ratio = (max(query_degree, cand_degree) + 1.0) / (min(query_degree, cand_degree) + 1.0)

        homophily_gap = 0.0
        if self.node_avg_homophily is not None:
            query_homophily = float(self.node_avg_homophily[query_node_id].item())
            cand_homophily = float(self.node_avg_homophily[cand_node_id].item())
            homophily_gap = abs(query_homophily - cand_homophily)

        return {
            "neighbor_cos": float(neighbor_cos),
            "degree_ratio": float(degree_ratio),
            "homophily_gap": float(homophily_gap),
            "query_degree": float(query_degree),
            "cand_degree": float(cand_degree),
        }

    def _passes_local_structure_check(self, query_node_id, entry, feat_cos, hamming_dist):
        if self.structure_check_mode == "sketch":
            return self._passes_topology_sketch_guard(query_node_id, entry, hamming_dist)

        cand_node_id = entry["node_id"]
        if cand_node_id == query_node_id:
            return True
        metrics = self._compute_local_structure_metrics(query_node_id, entry)
        neighbor_cos = metrics["neighbor_cos"]
        degree_ratio = metrics["degree_ratio"]
        homophily_gap = metrics["homophily_gap"]
        query_degree = metrics["query_degree"]
        cand_degree = metrics["cand_degree"]

        neighbor_tau = self.structure_neighbor_tau
        degree_ratio_max = self.structure_degree_ratio_max
        homophily_gap_max = self.structure_homophily_gap_max

        if hamming_dist <= 0:
            neighbor_tau -= 0.05
            degree_ratio_max += 0.5
            homophily_gap_max += 1.0
        if feat_cos >= self.second_stage_tau + 0.05:
            neighbor_tau -= 0.05
        if query_degree <= 1.0 or cand_degree <= 1.0:
            neighbor_tau -= 0.10
            degree_ratio_max += 1.0

        neighbor_tau = max(-1.0, neighbor_tau)
        return (
            neighbor_cos >= neighbor_tau
            and degree_ratio <= degree_ratio_max
            and homophily_gap <= homophily_gap_max
        )

    def _passes_topology_sketch_guard(self, query_node_id, entry, hamming_dist):
        cand_node_id = int(entry["node_id"])
        if cand_node_id == query_node_id:
            return True

        degree_bucket_gap = self.topology_degree_bucket_gap
        homophily_bucket_gap = self.topology_homophily_bucket_gap
        sketch_radius = self.topology_sketch_radius

        if hamming_dist <= 0:
            if degree_bucket_gap is not None:
                degree_bucket_gap += 1
            homophily_bucket_gap += 1
            sketch_radius += 1

        query_degree = float(self.node_total_degree[query_node_id].item())
        cand_degree = float(self.node_total_degree[cand_node_id].item())
        if query_degree <= 1.0 or cand_degree <= 1.0:
            if degree_bucket_gap is not None:
                degree_bucket_gap += 1
            homophily_bucket_gap += 1

        if degree_bucket_gap is not None and self.node_degree_bucket is not None:
            query_degree_bucket = int(self.node_degree_bucket[query_node_id].item())
            cand_degree_bucket = int(self.node_degree_bucket[cand_node_id].item())
            if abs(query_degree_bucket - cand_degree_bucket) > degree_bucket_gap:
                return False

        if self.node_homophily_bucket is not None:
            query_homophily_bucket = int(self.node_homophily_bucket[query_node_id].item())
            cand_homophily_bucket = int(self.node_homophily_bucket[cand_node_id].item())
            if abs(query_homophily_bucket - cand_homophily_bucket) > homophily_bucket_gap:
                return False

        if not self.enable_topology_sketch_guard or self.node_topology_sketch is None:
            return True

        query_hash = int(self.node_topology_sketch[query_node_id])
        cand_hash = int(self.node_topology_sketch[cand_node_id])
        topology_hamming = bin(query_hash ^ cand_hash).count("1")
        return topology_hamming <= sketch_radius

    def _get_bucket_population(self, h, route_idx=0):
        route_state = self._get_route_state(route_idx)
        return route_state["hash_population"].get(h, 0)

    def _should_guard_exact_bucket(self, h, route_idx=0):
        bucket_population = self._get_bucket_population(h, route_idx=route_idx)
        if bucket_population < self.exact_guard_min_bucket_size:
            return False
        if bucket_population >= self.exact_guard_large_bucket_size:
            return True
        route_bits = self.hash_route_bits[route_idx]
        return route_bits <= self.exact_guard_low_bits

    def _passes_exact_bucket_guard(self, accepted):
        if not accepted:
            return False

        top_item = accepted[0]
        strict_tau = min(
            1.0,
            self._required_route_cosine_tau(top_item["route_idx"], top_item["dist"]) + self.exact_guard_cosine_bonus,
        )
        if top_item["cos"] < strict_tau:
            return False

        if len(accepted) >= 2:
            runner_up = accepted[1]
            if (top_item["cos"] - runner_up["cos"]) < self.exact_guard_min_margin:
                return False
        return True

    def _candidate_sort_key(self, item):
        if self.hamming_only_acceptor:
            return (
                -int(item.get("winning_base_table_hit_count", 1)),
                -int(item.get("base_route_hit_count", 1)),
                -int(item.get("route_hit_count", 1)),
                item["dist"],
                -item["route_score"],
                -item["entry"]["timestamp"],
            )
        return (-item["route_score"], item["dist"], -item["cos"], -item["entry"]["timestamp"])

    def _required_cosine_tau(self, hamming_dist):
        if hamming_dist <= 0:
            return max(0.0, self.second_stage_tau - 0.10)
        if hamming_dist == 1:
            return max(0.0, self.second_stage_tau - 0.05)
        return self.second_stage_tau

    def _required_route_cosine_tau(self, route_idx, hamming_dist):
        base_tau = self._required_cosine_tau(hamming_dist)
        adjusted_tau = base_tau + self.route_accept_tau_offsets[route_idx]
        return min(1.0, max(0.0, adjusted_tau))

    def _find_exact_candidate_refs(self, query_hashes):
        candidate_refs = []
        for route_idx, query_hash in enumerate(query_hashes):
            route_state = self._get_route_state(route_idx)
            if route_state["memo_table"].get(query_hash):
                candidate_refs.append(
                    {
                        "route_idx": route_idx,
                        "hash": query_hash,
                        "query_hash": query_hash,
                    }
                )
        return candidate_refs

    def _collect_union_candidate_refs(self, query_hashes, allowed_r):
        candidate_refs = []
        seen_pairs = set()

        for route_idx, query_hash in enumerate(query_hashes):
            candidate_hashes = self.find_fuzzy_candidate_hashes_adaptive(
                query_hash,
                allowed_r,
                route_idx=route_idx,
                max_candidates=self.max_candidates_per_route,
            )
            for cand_hash in candidate_hashes:
                pair = (route_idx, cand_hash)
                if pair in seen_pairs:
                    continue
                candidate_refs.append(
                    {
                        "route_idx": route_idx,
                        "hash": cand_hash,
                        "query_hash": query_hash,
                    }
                )
                seen_pairs.add(pair)

        candidate_refs.sort(key=lambda item: (bin(item["query_hash"] ^ item["hash"]).count("1"), item["route_idx"]))
        return candidate_refs[: self.max_total_candidates]

    def _collect_scored_candidates(
        self,
        candidate_refs,
        query_feat,
        exclude_node_id=None,
        allowed_node_ids=None,
        prefer_fine_representative=False,
        prefer_main_fine_representative=False,
    ):
        best_by_node_route = {}
        if allowed_node_ids is not None and not isinstance(allowed_node_ids, set):
            allowed_node_ids = set(allowed_node_ids)
        for cand_ref in candidate_refs:
            route_idx = cand_ref["route_idx"]
            cand_hash = cand_ref["hash"]
            query_hash = cand_ref["query_hash"]
            route_state = self._get_route_state(route_idx)
            entries = route_state["memo_table"].get(cand_hash, [])
            if not entries:
                continue
            hash_dist = bin(query_hash ^ cand_hash).count("1")
            for entry in entries:
                node_id = entry["node_id"]
                if exclude_node_id is not None and node_id == exclude_node_id:
                    continue
                if allowed_node_ids is not None and node_id not in allowed_node_ids:
                    continue
                if self.hamming_only_acceptor:
                    cos = self._compute_hamming_match_score(route_idx, hash_dist)
                else:
                    cos = self._compute_entry_cosine(entry, query_feat)
                item = {
                    "route_idx": route_idx,
                    "route_name": self.hash_route_names[route_idx],
                    "route_bits": self.hash_route_bits[route_idx],
                    "base_route_idx": self.route_base_indices[route_idx],
                    "base_route_name": self.route_base_names[self.route_base_indices[route_idx]],
                    "route_weight": self.route_score_weights[route_idx],
                    "hash": cand_hash,
                    "entry": entry,
                    "cos": cos,
                    "route_score": cos * self.route_score_weights[route_idx],
                    "dist": hash_dist,
                }
                best_item = best_by_node_route.get((node_id, route_idx))
                if best_item is None or self._candidate_sort_key(item) < self._candidate_sort_key(best_item):
                    best_by_node_route[(node_id, route_idx)] = item

        aggregated_by_node = {}
        for (node_id, _route_idx), item in best_by_node_route.items():
            aggregated_by_node.setdefault(node_id, []).append(item)

        scored = []
        fine_bits_min = None
        if self.coarse_union_bits_max is not None:
            fine_bits_min = int(self.coarse_union_bits_max) + 1
        for route_items in aggregated_by_node.values():
            representative_items = route_items
            if fine_bits_min is not None:
                if prefer_main_fine_representative:
                    main_fine_route_items = [
                        item
                        for item in route_items
                        if int(item.get("base_route_idx", 0)) == 0
                        and int(item.get("route_bits", self.sketch_bits)) >= fine_bits_min
                    ]
                    if main_fine_route_items:
                        representative_items = main_fine_route_items
                    elif prefer_fine_representative:
                        fine_route_items = [
                            item
                            for item in route_items
                            if int(item.get("route_bits", self.sketch_bits)) >= fine_bits_min
                        ]
                        if fine_route_items:
                            representative_items = fine_route_items
                elif prefer_fine_representative:
                    fine_route_items = [
                        item
                        for item in route_items
                        if int(item.get("route_bits", self.sketch_bits)) >= fine_bits_min
                    ]
                    if fine_route_items:
                        representative_items = fine_route_items
            best_item = min(representative_items, key=self._candidate_sort_key)
            route_hit_count = len(route_items)
            route_names = tuple(sorted(item["route_name"] for item in route_items))
            base_route_idxs = tuple(sorted({int(item["base_route_idx"]) for item in route_items}))
            base_route_names = tuple(self.route_base_names[idx] for idx in base_route_idxs)
            winning_base_route_idx = int(best_item["base_route_idx"])
            max_base_route_idx = max(self.route_base_indices) if self.route_base_indices else -1
            base_route_table_hit_counts = [0] * (max_base_route_idx + 1)
            for item in route_items:
                base_route_table_hit_counts[int(item["base_route_idx"])] += 1
            winning_base_table_hit_count = sum(
                1 for item in route_items if int(item["base_route_idx"]) == winning_base_route_idx
            )
            winning_route_idx = int(best_item["route_idx"])
            winning_route_fine_support_count = 0
            cross_route_fine_support_count = 0
            winning_base_fine_support_count = 0
            main_route_fine_support_count = 0
            union_route_fine_support_count = 0
            coarse_head_support_count = 0
            fine_head_support_count = 0
            if fine_bits_min is not None or self.coarse_union_bits_max is not None:
                for item in route_items:
                    route_bits = int(item.get("route_bits", self.sketch_bits))
                    if fine_bits_min is not None and route_bits >= fine_bits_min:
                        fine_head_support_count += 1
                        if int(item["base_route_idx"]) == 0:
                            main_route_fine_support_count += 1
                        else:
                            union_route_fine_support_count += 1
                        if int(item["route_idx"]) == winning_route_idx:
                            winning_route_fine_support_count += 1
                        else:
                            cross_route_fine_support_count += 1
                        if int(item["base_route_idx"]) == winning_base_route_idx:
                            winning_base_fine_support_count += 1
                    if self.coarse_union_bits_max is not None and route_bits <= self.coarse_union_bits_max:
                        coarse_head_support_count += 1
            aggregate_item = dict(best_item)
            aggregate_item["route_hit_count"] = route_hit_count
            aggregate_item["route_names"] = route_names
            aggregate_item["route_idxs"] = tuple(sorted(item["route_idx"] for item in route_items))
            aggregate_item["base_route_hit_count"] = len(base_route_idxs)
            aggregate_item["base_route_idxs"] = base_route_idxs
            aggregate_item["base_route_names"] = base_route_names
            aggregate_item["base_route_table_hit_counts"] = tuple(base_route_table_hit_counts)
            aggregate_item["winning_base_route_idx"] = winning_base_route_idx
            aggregate_item["winning_base_route_name"] = self.route_base_names[winning_base_route_idx]
            aggregate_item["winning_base_table_hit_count"] = winning_base_table_hit_count
            aggregate_item["winning_route_fine_support_count"] = winning_route_fine_support_count
            aggregate_item["cross_route_fine_support_count"] = cross_route_fine_support_count
            aggregate_item["winning_base_fine_support_count"] = winning_base_fine_support_count
            aggregate_item["main_route_fine_support_count"] = main_route_fine_support_count
            aggregate_item["union_route_fine_support_count"] = union_route_fine_support_count
            aggregate_item["coarse_head_support_count"] = coarse_head_support_count
            aggregate_item["fine_head_support_count"] = fine_head_support_count
            aggregate_item["route_score"] = best_item["route_score"]
            scored.append(aggregate_item)
        scored.sort(key=self._candidate_sort_key)
        return scored

    def _select_vote_from_scored(
        self,
        scored,
        query_feat,
        query_node_id,
        apply_structure_check=True,
        exact_guard=False,
    ):
        if not scored:
            return None

        accepted = []
        max_structure_checks = self.max_structure_checks
        if max_structure_checks is None:
            max_structure_checks = max(self.vote_top_m * 4, 8)
        for item in scored[:max_structure_checks]:
            if not self.hamming_only_acceptor:
                required_tau = self._required_route_cosine_tau(item["route_idx"], item["dist"])
                relaxed_tau = max(0.0, required_tau - self.vote_relax_margin)
                if item["cos"] < relaxed_tau:
                    continue
            if apply_structure_check and not self.disable_structure_check:
                self.stats["structure_checked"] += 1
                structure_feat_score = 0.0 if self.hamming_only_acceptor else item["cos"]
                if not self._passes_local_structure_check(
                    query_node_id,
                    item["entry"],
                    structure_feat_score,
                    item["dist"],
                ):
                    self.stats["structure_reject"] += 1
                    continue
            score_ok, _score_decision = self._score_gate_allows(query_node_id, item)
            if not score_ok:
                continue
            accepted.append(item)

        if not accepted:
            return None

        top_item = accepted[0]
        top_required_tau = self._required_route_cosine_tau(top_item["route_idx"], top_item["dist"])
        route_vote_count = sum(1 for item in accepted if item["route_idx"] == top_item["route_idx"])
        if route_vote_count < self.route_min_accept_votes[top_item["route_idx"]]:
            return None
        if int(top_item.get("base_route_hit_count", 0)) < self.min_base_route_hits:
            return None
        winning_base_route_idx = int(top_item.get("winning_base_route_idx", 0))
        winning_base_table_hit_count = int(top_item.get("winning_base_table_hit_count", 1))
        if winning_base_table_hit_count < self.route_min_support_hits[winning_base_route_idx]:
            return None
        if not self.hamming_only_acceptor and top_item["cos"] < top_required_tau and len(accepted) < 2:
            return None
        if exact_guard and not self.hamming_only_acceptor and not self._passes_exact_bucket_guard(accepted):
            self.stats["exact_guard_reject"] += 1
            return None

        return {
            "best_route_idx": top_item["route_idx"],
            "best_route_name": top_item["route_name"],
            "best_hash": top_item["hash"],
            "best_entry": top_item["entry"],
            "best_cos": top_item["cos"],
            "best_dist": top_item["dist"],
            "voted_emb": top_item["entry"]["cached_emb"],
        }

    def _aggregate_candidate_vote(
        self,
        candidate_refs,
        query_feat,
        query_node_id,
        apply_structure_check=True,
        exact_guard=False,
    ):
        scored = self._collect_scored_candidates(candidate_refs, query_feat, exclude_node_id=query_node_id)
        return self._select_vote_from_scored(
            scored,
            query_feat,
            query_node_id,
            apply_structure_check=apply_structure_check,
            exact_guard=exact_guard,
        )

    def query_full_batch(self, hash_features, verify_features, oracle_embs):
        num_nodes = verify_features.size(0)
        self._reset_route_caches()
        self.stats = {
            "total_queries": 0,
            "reuse": 0,
            "exact_reuse": 0,
            "fuzzy_reuse": 0,
            "fuzzy": 0,
            "computed": 0,
            "structure_checked": 0,
            "structure_reject": 0,
            "exact_guarded": 0,
            "exact_guard_reject": 0,
            "score_checked": 0,
            "score_reject": 0,
            "score_reject_hub_protect": 0,
            "score_reject_rare_leaf": 0,
            "score_reject_risk": 0,
            "score_risk_sum": 0.0,
            "score_sensitivity_sum": 0.0,
            "score_count": 0,
            "quant_checked": 0,
            "quant_int4": 0,
            "quant_int8": 0,
            "full_precision": 0,
            "protected": 0,
            "quant_risk_sum": 0.0,
            "quant_sensitivity_sum": 0.0,
        }

        print(f"[Adaptive] Starting Full Batch Simulation on {num_nodes} nodes...")

        hash_feature_routes = self._normalize_hash_feature_routes(hash_features)
        all_hashes = [
            self._compute_route_fingerprint(route_features, route_idx)
            for route_idx, route_features in enumerate(hash_feature_routes)
        ]

        final_embs_list = [None] * num_nodes
        hits_list = [False] * num_nodes
        indices_compute = []

        for node_idx in range(num_nodes):
            query_hashes = [route_hashes[node_idx] for route_hashes in all_hashes]
            allowed_r = self.node_policies[node_idx].item()

            exact_candidate_refs = self._find_exact_candidate_refs(query_hashes)
            if exact_candidate_refs:
                exact_guard = any(
                    self._should_guard_exact_bucket(ref["hash"], route_idx=ref["route_idx"])
                    for ref in exact_candidate_refs
                )
                if exact_guard:
                    self.stats["exact_guarded"] += 1
                vote_result = self._aggregate_candidate_vote(
                    exact_candidate_refs,
                    verify_features[node_idx],
                    node_idx,
                    apply_structure_check=exact_guard or self.hamming_only_acceptor,
                    exact_guard=exact_guard,
                )
                if vote_result is not None:
                    best_entry = vote_result["best_entry"]
                    self._touch_route_entry(vote_result["best_route_idx"], vote_result["best_hash"], best_entry)
                    final_embs_list[node_idx] = vote_result["voted_emb"]
                    hits_list[node_idx] = True
                    self.stats["reuse"] += 1
                    self.stats["exact_reuse"] += 1
                    continue

            candidate_refs = self._collect_union_candidate_refs(query_hashes, allowed_r)
            vote_result = self._aggregate_candidate_vote(
                candidate_refs,
                verify_features[node_idx],
                node_idx,
                apply_structure_check=True,
            )
            if vote_result is not None:
                best_entry = vote_result["best_entry"]
                self._touch_route_entry(vote_result["best_route_idx"], vote_result["best_hash"], best_entry)
                final_embs_list[node_idx] = vote_result["voted_emb"]
                hits_list[node_idx] = True
                self.stats["reuse"] += 1
                self.stats["fuzzy_reuse"] += 1
                self.stats["fuzzy"] += 1
                continue

            indices_compute.append(node_idx)
            real_emb = oracle_embs[node_idx]
            produced_emb, _quant_decision = self._materialize_computed_embedding(node_idx, real_emb)
            entry = {
                "node_id": int(node_idx),
                "cheap_feat": verify_features[node_idx].detach().clone(),
                "cached_emb": produced_emb.detach().clone(),
                "timestamp": self._time_counter,
            }
            self._time_counter += 1
            self._cache_computed_entry(query_hashes, entry)
            final_embs_list[node_idx] = produced_emb

        self.stats["total_queries"] = num_nodes
        self.stats["computed"] = len(indices_compute)
        self.stats["reuse_denominator"] = num_nodes
        self.stats["accounted_queries"] = self.stats["reuse"] + self.stats["computed"]
        self.stats["reuse_consistency_ok"] = self.stats["reuse"] == self.stats["exact_reuse"] + self.stats["fuzzy_reuse"]
        self.stats["query_consistency_ok"] = self.stats["accounted_queries"] == num_nodes
        self.stats["avg_score_risk"] = (
            self.stats["score_risk_sum"] / self.stats["score_count"]
            if self.stats["score_count"] > 0
            else 0.0
        )
        self.stats["avg_score_sensitivity"] = (
            self.stats["score_sensitivity_sum"] / self.stats["score_count"]
            if self.stats["score_count"] > 0
            else 0.0
        )
        self.stats["quantized"] = self.stats["quant_int4"] + self.stats["quant_int8"]
        self.stats["execution_accounted_queries"] = (
            self.stats["reuse"]
            + self.stats["quant_int4"]
            + self.stats["quant_int8"]
            + self.stats["full_precision"]
            + self.stats["protected"]
        )
        self.stats["execution_consistency_ok"] = self.stats["execution_accounted_queries"] == num_nodes
        self.stats["avg_quant_risk"] = (
            self.stats["quant_risk_sum"] / self.stats["quant_checked"]
            if self.stats["quant_checked"] > 0
            else 0.0
        )
        self.stats["avg_quant_sensitivity"] = (
            self.stats["quant_sensitivity_sum"] / self.stats["quant_checked"]
            if self.stats["quant_checked"] > 0
            else 0.0
        )

        final_embs = torch.stack(final_embs_list, dim=0)
        hits_tensor = torch.tensor(hits_list, dtype=torch.bool, device=self.device)
        return final_embs, hits_tensor
