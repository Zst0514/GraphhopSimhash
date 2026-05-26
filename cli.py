import argparse

from .features import HASH_VIEW_PRESETS
from .routing import list_retrieval_route_tags
from .runner import run_adaptive_simulation

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compact paper-style runner for graph hash reuse experiments."
    )

    parser.add_argument("--datasets", nargs="+", default=["cora"], choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--experiment_suite",
        type=str,
        default="single",
        choices=[
            "single",
            "score_ablation",
            "quant_ablation",
            "real_quant_ablation",
            "reuse_real_quant",
            "residual_reuse",
            "graph_eager_token",
            "token_compaction",
            "ffn_channel_gating",
            "hierarchical_encoder",
        ],
        help=(
            "Run one config, score/quant ablations, real quantization, joint reuse+real-quantization, "
            "residual reuse validation, graph-eager token routing, token compaction validation, "
            "FFN channel-gating validation, or full hierarchical encoder validation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Unified base seed. Each run uses --seed + run_idx unless a specific seed override is provided.",
    )
    parser.add_argument("--max_test", type=int, default=None, help="Override test split size for faster debugging")

    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--sketch_bits", type=int, default=14)
    parser.add_argument(
        "--hash_view",
        type=str,
        default="self_1hop_2hop",
        choices=sorted(HASH_VIEW_PRESETS.keys()),
        help="Primary retrieval view preset; overridden by --hash_mix_weights for the main route",
    )
    parser.add_argument(
        "--hash_mix_weights",
        type=float,
        nargs=3,
        default=[0.3, 0.7, 0.0],
        metavar=("SELF_W", "HOP1_W", "HOP2_W"),
        help="Main-route self/1hop/2hop mixing weights. Default mainline is 0.3 0.7 0.0",
    )
    parser.add_argument(
        "--union_hash_views",
        nargs="+",
        default=[],
        choices=sorted(HASH_VIEW_PRESETS.keys()),
        help="Optional auxiliary retrieval views unioned with the primary view",
    )
    parser.add_argument("--cosine_tau", type=float, default=0.59)

    parser.add_argument("--cache_size", type=int, default=None)
    parser.add_argument("--memo_k", type=int, default=3)
    parser.add_argument("--vote_top_m", type=int, default=4)
    parser.add_argument("--vote_relax_margin", type=float, default=0.05)

    parser.add_argument("--learned_hash_projection", dest="learned_hash_projection", action="store_true")
    parser.add_argument("--no_learned_hash_projection", dest="learned_hash_projection", action="store_false")
    parser.set_defaults(learned_hash_projection=True)
    parser.add_argument("--learned_hash_dim", type=int, default=256)
    parser.add_argument("--learned_hash_epochs", type=int, default=80)
    parser.add_argument("--learned_hash_lr", type=float, default=1e-3)
    parser.add_argument("--learned_hash_weight_decay", type=float, default=1e-4)
    parser.add_argument("--learned_hash_batch_size", type=int, default=512)
    parser.add_argument(
        "--learned_hash_supervision",
        type=str,
        default="train_val",
        choices=["train", "train_val"],
    )
    parser.add_argument("--learned_hash_supervision_limit", type=int, default=2048)
    parser.add_argument("--learned_hash_topk", type=int, default=48)
    parser.add_argument("--learned_hash_pos_per_anchor", type=int, default=4)
    parser.add_argument("--learned_hash_neg_per_anchor", type=int, default=8)
    parser.add_argument("--learned_hash_pos_tau", type=float, default=0.95)
    parser.add_argument("--learned_hash_neg_tau", type=float, default=0.85)
    parser.add_argument("--learned_hash_neg_margin", type=float, default=0.30)
    parser.add_argument("--learned_hash_balance_lambda", type=float, default=0.05)
    parser.add_argument(
        "--shared_hash_projection_heads",
        action="store_true",
        help="Train one learned projection per base route and reuse it across hash heads",
    )
    parser.add_argument(
        "--hash_heads_per_route",
        "--hash_tables_per_route",
        dest="hash_heads_per_route",
        type=int,
        default=4,
        help="Number of learned hash heads per base retrieval view",
    )
    parser.add_argument(
        "--hash_head_seed",
        "--hash_table_seed",
        dest="hash_head_seed",
        type=int,
        default=12345,
        help="Base seed for learned hash heads. Default 12345 preserves the original experiment setting.",
    )
    parser.add_argument(
        "--hash_head_bits",
        type=int,
        nargs="+",
        default=None,
        help="Optional per-head hash bit schedule, e.g. 14 16 14 12",
    )
    parser.add_argument(
        "--main_hash_head_bits",
        type=int,
        nargs="+",
        default=[16, 16, 16, 16],
        help="Optional main-route head bit schedule. Overrides --hash_head_bits for the primary route",
    )
    parser.add_argument(
        "--union_hash_head_bits",
        type=int,
        nargs="+",
        default=None,
        help="Optional union-route head bit schedule. Overrides --hash_head_bits for auxiliary routes",
    )
    parser.add_argument(
        "--enable_topology_retrieval_route",
        action="store_true",
        help="Append a topology-context retrieval route so semantic and topology chunks can be matched together",
    )
    parser.add_argument(
        "--topology_hash_head_bits",
        type=int,
        nargs="+",
        default=[16, 16],
        help="Optional topology-route head bit schedule. Used only when --enable_topology_retrieval_route is set",
    )

    parser.add_argument("--route_score_weights", type=float, nargs="+", default=None)
    parser.add_argument("--union_route_weight", type=float, default=0.94)
    parser.add_argument("--table_route_weight_decay", type=float, default=0.95)
    parser.add_argument("--route_accept_tau_offsets", type=float, nargs="+", default=None)
    parser.add_argument("--union_accept_tau_bonus", type=float, default=0.01)
    parser.add_argument("--route_min_accept_votes", type=int, nargs="+", default=None)
    parser.add_argument("--union_min_accept_votes", type=int, default=1)
    parser.add_argument("--route_min_support_hits", type=int, nargs="+", default=[2])
    parser.add_argument("--union_min_support_hits", type=int, default=1)
    parser.add_argument(
        "--min_base_route_hits",
        type=int,
        default=1,
        help="Require the accepted candidate to be supported by at least this many distinct route groups",
    )

    parser.add_argument("--max_candidates_per_route", type=int, default=96)
    parser.add_argument("--max_total_candidates", type=int, default=640)
    parser.add_argument("--max_structure_checks", type=int, default=16)

    parser.add_argument("--coarse_union_bits_max", type=int, default=None)

    parser.add_argument("--structure_neighbor_tau", type=float, default=0.45)
    parser.add_argument("--structure_degree_ratio_max", type=float, default=4.0)
    parser.add_argument("--structure_homophily_gap_max", type=float, default=None)
    parser.add_argument(
        "--structure_check_mode",
        type=str,
        default="sketch",
        choices=["cosine", "sketch"],
    )
    parser.add_argument(
        "--enable_homophily_bucket_guard",
        action="store_true",
        help="Enable homophily-bucket filtering in sketch structure checks. Default is disabled.",
    )
    parser.add_argument(
        "--enable_topology_sketch_guard",
        action="store_true",
        help="Enable topology-sketch Hamming filtering in the backend guard.",
    )
    parser.add_argument("--topology_sketch_bits", type=int, default=16)
    parser.add_argument("--topology_sketch_radius", type=int, default=5)
    parser.add_argument("--topology_degree_bucket_gap", type=int, default=None)
    parser.add_argument("--topology_homophily_bins", type=int, default=8)
    parser.add_argument("--topology_homophily_bucket_gap", type=int, default=2)
    parser.add_argument(
        "--topology_sketch_seed",
        type=int,
        default=None,
        help="Advanced override for topology sketch generation. If omitted, uses the effective run seed.",
    )
    parser.add_argument(
        "--disable_structure_check",
        action="store_true",
        help="Ablate local topology checks during candidate acceptance",
    )

    parser.add_argument("--exact_guard_low_bits", type=int, default=16)
    parser.add_argument("--exact_guard_min_bucket_size", type=int, default=2)
    parser.add_argument("--exact_guard_large_bucket_size", type=int, default=4)
    parser.add_argument("--exact_guard_min_margin", type=float, default=0.03)
    parser.add_argument("--exact_guard_cosine_bonus", type=float, default=0.02)

    parser.add_argument(
        "--hamming_only_acceptor",
        action="store_true",
        help="Use only Hamming distance and multi-head support for candidate acceptance; disables cheap-feature cosine reranking",
    )
    parser.add_argument(
        "--enable_score_gate",
        dest="disable_score_gate",
        action="store_false",
        help="Enable the degree/context/rare-leaf risk gate after retrieval.",
    )
    parser.add_argument(
        "--disable_score_gate",
        dest="disable_score_gate",
        action="store_true",
        help="Disable degree/context/rare-leaf risk gate after retrieval.",
    )
    parser.set_defaults(disable_score_gate=True)
    parser.add_argument("--score_reuse_threshold", type=int, default=45)
    parser.add_argument("--score_hub_threshold", type=int, default=12)
    parser.add_argument("--score_rare_threshold", type=int, default=10)
    parser.add_argument(
        "--score_protect_hub_exact",
        action="store_true",
        help="Also reject exact hash reuse for high-propagation nodes",
    )
    parser.add_argument(
        "--allow_hub_fuzzy",
        action="store_true",
        help="Do not hard-block fuzzy reuse for high-propagation nodes; leave them to the risk threshold.",
    )
    parser.add_argument(
        "--allow_rare_fuzzy",
        action="store_true",
        help="Allow fuzzy reuse for low-degree rare nodes; by default it is blocked",
    )
    parser.add_argument(
        "--disable_score_support_discount",
        action="store_true",
        help="Do not reduce approximation error when a candidate is supported by multiple hash heads.",
    )
    parser.add_argument(
        "--score_rare_gate_mode",
        type=str,
        default="support",
        choices=["hard", "support", "risk"],
        help="How to handle low-degree rare fuzzy candidates: old hard block, support-aware block, or risk-only.",
    )
    parser.add_argument("--score_rare_min_dist", type=int, default=2)
    parser.add_argument("--score_rare_min_route_hits", type=int, default=2)
    parser.add_argument("--score_rare_min_base_hits", type=int, default=2)
    parser.add_argument("--score_pair_confidence_discount", type=int, default=1)
    parser.add_argument("--score_pair_confidence_max_dist", type=int, default=1)
    parser.add_argument("--score_pair_confidence_min_route_hits", type=int, default=2)
    parser.add_argument("--score_pair_confidence_min_base_hits", type=int, default=2)
    parser.add_argument("--score_pair_confidence_min_cos_margin", type=float, default=0.02)
    parser.add_argument("--score_rarity_bits", type=int, default=16)
    parser.add_argument("--score_rarity_seed", type=int, default=98765)
    parser.add_argument("--score_propagation_weight", type=int, default=3)
    parser.add_argument("--score_graph_context_weight", type=int, default=2)
    parser.add_argument("--score_low_unique_weight", type=int, default=2)
    parser.add_argument(
        "--enable_quant_policy",
        action="store_true",
        help="After reuse misses, choose INT4/INT8/protected execution using the same sensitivity score.",
    )
    parser.add_argument("--quant_int4_threshold", type=int, default=90)
    parser.add_argument("--quant_int8_threshold", type=int, default=45)
    parser.add_argument("--quant_int4_error", type=int, default=3)
    parser.add_argument("--quant_int8_error", type=int, default=1)
    parser.add_argument("--quant_int4_bits", type=int, default=4)
    parser.add_argument("--quant_int8_bits", type=int, default=8)
    parser.add_argument("--quant_tser_propagation_weight", type=float, default=4.0)
    parser.add_argument("--quant_tser_graph_context_weight", type=float, default=1.0)
    parser.add_argument("--quant_tser_low_unique_weight", type=float, default=0.0)
    parser.add_argument(
        "--quant_error_bias",
        type=float,
        default=1.0,
        help="Bias added to quantized INT4 error for error-aware TopK ranking.",
    )
    parser.add_argument(
        "--quant_error_rank_source",
        type=str,
        default="continuous",
        choices=["continuous", "quantized"],
        help="Error signal used by DegreeErrorTopK/TSERErrorTopK ranking.",
    )
    parser.add_argument("--residual_rank", type=int, default=32)
    parser.add_argument("--residual_epochs", type=int, default=200)
    parser.add_argument("--residual_lr", type=float, default=1e-3)
    parser.add_argument("--residual_weight_decay", type=float, default=1e-4)
    parser.add_argument("--residual_l2", type=float, default=1e-4)
    parser.add_argument(
        "--residual_embedding_source",
        type=str,
        default="data_x",
        choices=["data_x", "real_quant_fp"],
        help="Target embedding pool for residual reuse. data_x keeps the cached ST/OFA features; real_quant_fp uses --real_quant_* FP pool.",
    )
    parser.add_argument(
        "--residual_embedding_path",
        type=str,
        default=None,
        help="Optional explicit target embedding path for residual reuse. Overrides --real_quant_fp_path when set.",
    )
    parser.add_argument(
        "--residual_alpha",
        type=float,
        default=-1.0,
        help="Residual application strength. Use a negative value to select from --residual_alpha_grid on val.",
    )
    parser.add_argument(
        "--residual_alpha_grid",
        nargs="+",
        type=float,
        default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0],
        help="Candidate residual strengths when --residual_alpha is negative.",
    )
    parser.add_argument(
        "--residual_min_dist",
        type=float,
        default=1.0,
        help="Only apply residual correction to reuse hits with Hamming distance at least this value.",
    )
    parser.add_argument(
        "--residual_direct_threshold",
        type=float,
        default=-1.0,
        help="If non-negative, hits with reuse risk at or below this threshold stay direct; only higher-risk hits are corrected.",
    )
    parser.add_argument(
        "--residual_anchor_mode",
        type=str,
        default="cam",
        choices=["cam", "random"],
        help="Use CAM-selected anchors or replace hit anchors with random nodes for ablation.",
    )
    parser.add_argument(
        "--residual_min_route_hits",
        type=int,
        default=1,
        help="Only train/apply residual correction to hits supported by at least this many route heads.",
    )
    parser.add_argument(
        "--residual_min_base_hits",
        type=int,
        default=1,
        help="Only train/apply residual correction to hits supported by at least this many base tables.",
    )
    parser.add_argument(
        "--residual_hard_min_support_hits",
        type=int,
        default=-1,
        help="If >0, accepted hits with at least this many head/table hits stay as hard direct reuse.",
    )
    parser.add_argument(
        "--residual_soft_min_support_hits",
        type=int,
        default=-1,
        help="If >0, collect hits with at least this many head/table hits; hits below hard threshold use residual correction.",
    )
    parser.add_argument("--residual_max_train_pairs", type=int, default=4096)
    parser.add_argument(
        "--residual_train_split",
        type=str,
        default="train_val",
        choices=["train", "train_val", "all_hits"],
        help="Reuse-hit nodes used to fit the low-rank residual adapter.",
    )
    parser.add_argument(
        "--internal_split_calibration",
        "--enable_internal_split_calibration",
        dest="internal_split_calibration",
        action="store_true",
        help="Build a high-bit/low-bit node calibration split before the run.",
    )
    parser.add_argument(
        "--disable_internal_split_calibration",
        dest="internal_split_calibration",
        action="store_false",
        help="Disable internal split calibration.",
    )
    parser.set_defaults(internal_split_calibration=False)
    parser.add_argument(
        "--internal_split_calibration_only",
        action="store_true",
        help="Only build and save internal split calibration reports, then exit.",
    )
    parser.add_argument("--internal_calib_samples", type=int, default=512)
    parser.add_argument("--internal_calib_high_ratio", type=float, default=0.5)
    parser.add_argument(
        "--internal_split_priority",
        type=str,
        default="degree",
        choices=["degree", "tser", "random", "bottom_degree"],
        help="Node priority used to assign the high-bit calibration group.",
    )
    parser.add_argument("--internal_split_topk_ratio", type=float, default=0.10)
    parser.add_argument(
        "--internal_split_mass_target",
        type=float,
        default=-1.0,
        help="If >= 0, choose the smallest high-bit set whose priority mass reaches this value.",
    )
    parser.add_argument(
        "--internal_calib_strategy",
        type=str,
        default="topology",
        choices=["uniform", "topology"],
        help="Sampling strategy used separately inside the high-bit and low-bit groups.",
    )
    parser.add_argument(
        "--internal_calib_bucket_mode",
        type=str,
        default="degree_quantile",
        choices=["degree_quantile", "degree_topk"],
    )
    parser.add_argument("--internal_calib_bucket_count", type=int, default=4)
    parser.add_argument("--internal_calib_bucket_topk_ratio", type=float, default=-1.0)
    parser.add_argument("--internal_calib_uniform_ratio", type=float, default=0.5)
    parser.add_argument("--internal_calib_high_a_bit", type=int, default=8)
    parser.add_argument("--internal_calib_low_a_bit", type=int, default=4)
    parser.add_argument(
        "--internal_calib_report_dir",
        type=str,
        default=None,
        help="Directory for internal split calibration JSON reports. Defaults to the dataset log directory.",
    )
    parser.add_argument("--real_quant_model_name", type=str, default="llama2_7b")
    parser.add_argument("--real_quant_fp_tag", type=str, default="FP16")
    parser.add_argument("--real_quant_int8_tag", type=str, default="INT8")
    parser.add_argument("--real_quant_int4_tag", type=str, default="INT4")
    parser.add_argument("--real_quant_fp_path", type=str, default=None)
    parser.add_argument("--real_quant_int8_path", type=str, default=None)
    parser.add_argument("--real_quant_int4_path", type=str, default=None)
    parser.add_argument(
        "--graph_eager_reference_tag",
        type=str,
        default="W4A16",
        help="Reference full-sequence embedding tag for graph-eager token routing.",
    )
    parser.add_argument(
        "--graph_eager_full_tag",
        type=str,
        default="W4A8",
        help="Full-sequence W4A8 embedding tag for graph-eager token routing.",
    )
    parser.add_argument(
        "--graph_eager_token_tag_prefix",
        type=str,
        default="W4A8_S",
        help="Prefix for shortened-token pools, e.g. W4A8_S128/W4A8_S256.",
    )
    parser.add_argument(
        "--graph_eager_token_lengths",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Shortened token budgets to load for graph-eager routing.",
    )
    parser.add_argument(
        "--graph_eager_full_length",
        type=int,
        default=512,
        help="Full token length used by the graph-eager token cost model.",
    )
    parser.add_argument(
        "--graph_eager_cost_scale",
        type=float,
        default=0.50,
        help="Cost of full-sequence W4A8 relative to full FP encoder cost.",
    )
    parser.add_argument(
        "--graph_eager_attn_weight",
        type=float,
        default=0.35,
        help="Attention fraction in the token-length cost model.",
    )
    parser.add_argument(
        "--graph_eager_ffn_weight",
        type=float,
        default=0.65,
        help="FFN/projection fraction in the token-length cost model.",
    )
    parser.add_argument(
        "--graph_eager_full_ratio",
        type=float,
        default=0.20,
        help="Ratio routed to full-sequence W4A8 in graph-eager budget policies.",
    )
    parser.add_argument(
        "--graph_eager_mid_ratio",
        type=float,
        default=0.30,
        help="Ratio routed to the largest shortened-token pool. The rest use the shortest pool.",
    )
    parser.add_argument(
        "--graph_eager_predictor_calib_samples",
        type=int,
        default=512,
        help="Calibration nodes used to fit the graph-eager damage predictor.",
    )
    parser.add_argument(
        "--graph_eager_predictor_ridge",
        type=float,
        default=1e-2,
        help="Ridge regularization for the graph-eager linear predictor.",
    )
    parser.add_argument(
        "--graph_eager_predictor_target",
        type=str,
        default="embedding",
        choices=["embedding", "margin"],
        help="Predict shortened-token embedding damage or downstream logit margin damage.",
    )
    parser.add_argument("--token_compaction_reference_tag", type=str, default="W4A16")
    parser.add_argument("--token_compaction_full_tag", type=str, default="W4A8")
    parser.add_argument(
        "--token_compaction_tags",
        type=str,
        nargs="+",
        default=["W4A8_S128", "W4A8_S128_RANDOM", "W4A8_S128_TFIDF", "W4A8_S128_GRAPHCTX"],
        help="Embedding tags to compare for fixed-budget token/chunk compaction.",
    )
    parser.add_argument(
        "--token_compaction_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional display names matching --token_compaction_tags.",
    )
    parser.add_argument(
        "--token_compaction_length",
        type=int,
        default=128,
        help="Token budget length used for cost reporting in token_compaction.",
    )
    parser.add_argument("--ffn_gating_reference_tag", type=str, default="FP16")
    parser.add_argument("--ffn_gating_full_tag", type=str, default="W4A8")
    parser.add_argument(
        "--ffn_gating_tags",
        type=str,
        nargs="+",
        default=["W4A8_FFN50"],
        help="Generated embedding tags for FFN channel-gated pools.",
    )
    parser.add_argument(
        "--ffn_gating_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional display names matching --ffn_gating_tags.",
    )
    parser.add_argument(
        "--ffn_gating_keep_ratios",
        type=float,
        nargs="+",
        default=[0.5],
        help="FFN channel keep ratios matching --ffn_gating_tags.",
    )
    parser.add_argument(
        "--ffn_gating_route_ratios",
        type=float,
        nargs="+",
        default=[0.5],
        help="Fractions of nodes routed to the gated FFN path in graph-aware routing policies.",
    )
    parser.add_argument(
        "--ffn_gating_cost_scale",
        type=float,
        default=0.50,
        help="Cost of full W4A8 encoder relative to full FP encoder cost.",
    )
    parser.add_argument(
        "--ffn_gating_attn_weight",
        type=float,
        default=0.35,
        help="Non-FFN fraction that remains unchanged under FFN channel gating.",
    )
    parser.add_argument(
        "--ffn_gating_ffn_weight",
        type=float,
        default=0.65,
        help="FFN fraction scaled by the channel keep ratio.",
    )
    parser.add_argument("--hierarchical_reference_tag", type=str, default="FP16")
    parser.add_argument("--hierarchical_full_tag", type=str, default="W4A8")
    parser.add_argument("--hierarchical_gated_tag", type=str, default="W4A8_FFN75")
    parser.add_argument(
        "--hierarchical_router_reference",
        type=str,
        default="data_x",
        choices=["data_x", "execution_reference"],
        help=(
            "Supervision source for learned hash routing in hierarchical_encoder. "
            "data_x matches the original clean OFA/ST graph-feature route supervision; "
            "execution_reference uses --hierarchical_reference_tag."
        ),
    )
    parser.add_argument("--hierarchical_gated_name", type=str, default="FFN75")
    parser.add_argument("--hierarchical_gated_keep_ratio", type=float, default=0.75)
    parser.add_argument("--hierarchical_gated_route_ratio", type=float, default=0.40)
    parser.add_argument(
        "--hierarchical_gated_route_policy",
        type=str,
        default="tser",
        choices=["degree", "tser", "random"],
        help="Policy for selecting hash-miss nodes that use the FFN-gated encoder path.",
    )
    parser.add_argument(
        "--hierarchical_residual_cost",
        type=float,
        default=0.005,
        help="Tiny relative cost charged for fuzzy reuse residual correction per corrected node.",
    )
    parser.add_argument(
        "--disable_real_quant_autogen",
        action="store_true",
        help="For reuse_real_quant, use existing real-quant pools instead of regenerating and overwriting them.",
    )
    parser.add_argument(
        "--reuse_real_quant_allfp_only",
        action="store_true",
        help="For reuse_real_quant sweeps, report only the AllFP miss path so the run isolates hash reuse.",
    )
    parser.add_argument(
        "--real_quant_policy_suite",
        type=str,
        default="standard",
        choices=["standard", "w4a8_budget"],
        help="Policy set for real-quant ablation. w4a8_budget matches W4A4/W4A8/FP table-style comparisons.",
    )
    parser.add_argument(
        "--real_quant_error_space",
        type=str,
        default="encoded",
        choices=["raw", "encoded"],
        help="Compute policy quantization error in raw LLM-embedding space or after the trained GNN input projection.",
    )
    parser.add_argument(
        "--real_quant_error_norm",
        type=float,
        default=0.20,
        help="Cosine-error value that maps to quantized error 15 for real quant policy.",
    )
    parser.add_argument("--real_quant_int4_threshold", type=int, default=120)
    parser.add_argument("--real_quant_int8_threshold", type=int, default=80)
    parser.add_argument(
        "--real_quant_fp_ratio",
        type=float,
        default=0.10,
        help="FP/protected ratio for DegreeTopK/TSERTopK and cascade real-quant policies.",
    )
    parser.add_argument(
        "--real_quant_int8_ratio",
        type=float,
        default=0.20,
        help="INT8 ratio for DegreeCascade/TSERCascade. Tail nodes use INT4.",
    )
    parser.add_argument(
        "--real_quant_tail_precision",
        type=str,
        default="int4",
        choices=["int4", "int8"],
        help="Tail precision for TopK real-quant policies.",
    )
    parser.add_argument(
        "--controller_seed",
        type=int,
        default=None,
        help="Advanced override for hash controller init. If omitted, uses the effective run seed.",
    )
    parser.add_argument(
        "--standard_eval_baseline",
        action="store_true",
        help="Use eval-mode validation when selecting the baseline checkpoint. Default keeps legacy GraphAdaptiveMask behavior.",
    )
    parser.add_argument("--llm_name", type=str, default="ST")
    parser.add_argument("--emb_dim", type=int, default=768)
    return parser


def validate_args(parser, args):
    if args.sketch_bits <= 0:
        parser.error("--sketch_bits must be a positive integer")
    if args.radius < 0 or args.radius > args.sketch_bits:
        parser.error("--radius must be between 0 and --sketch_bits")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.max_test is not None and args.max_test <= 0:
        parser.error("--max_test must be a positive integer")
    if args.memo_k <= 0:
        parser.error("--memo_k must be a positive integer")
    if args.vote_top_m <= 0:
        parser.error("--vote_top_m must be a positive integer")
    if args.vote_relax_margin < 0:
        parser.error("--vote_relax_margin must be non-negative")
    if len(args.hash_mix_weights) != 3:
        parser.error("--hash_mix_weights must provide exactly 3 values")
    if sum(float(weight) for weight in args.hash_mix_weights) <= 0:
        parser.error("--hash_mix_weights must sum to a positive value")
    if args.learned_hash_dim <= 0:
        parser.error("--learned_hash_dim must be positive")
    if args.learned_hash_epochs <= 0:
        parser.error("--learned_hash_epochs must be positive")
    if args.learned_hash_lr <= 0:
        parser.error("--learned_hash_lr must be positive")
    if args.learned_hash_weight_decay < 0:
        parser.error("--learned_hash_weight_decay must be non-negative")
    if args.learned_hash_batch_size <= 0:
        parser.error("--learned_hash_batch_size must be positive")
    if args.learned_hash_supervision_limit is not None and args.learned_hash_supervision_limit <= 0:
        parser.error("--learned_hash_supervision_limit must be positive")
    if args.learned_hash_topk <= 0:
        parser.error("--learned_hash_topk must be positive")
    if args.learned_hash_pos_per_anchor <= 0:
        parser.error("--learned_hash_pos_per_anchor must be positive")
    if args.learned_hash_neg_per_anchor <= 0:
        parser.error("--learned_hash_neg_per_anchor must be positive")
    if not (0.0 <= args.learned_hash_pos_tau <= 1.0):
        parser.error("--learned_hash_pos_tau must be within [0, 1]")
    if not (0.0 <= args.learned_hash_neg_tau <= 1.0):
        parser.error("--learned_hash_neg_tau must be within [0, 1]")
    if args.learned_hash_neg_margin < 0 or args.learned_hash_neg_margin > 1.0:
        parser.error("--learned_hash_neg_margin must be within [0, 1]")
    if args.learned_hash_balance_lambda < 0:
        parser.error("--learned_hash_balance_lambda must be non-negative")
    if args.hash_heads_per_route <= 0:
        parser.error("--hash_heads_per_route must be positive")
    if args.hash_head_bits is not None:
        if len(args.hash_head_bits) not in (1, args.hash_heads_per_route):
            parser.error(f"--hash_head_bits expects 1 or {args.hash_heads_per_route} values")
        if any(bit <= 0 for bit in args.hash_head_bits):
            parser.error("--hash_head_bits values must be positive")
    if args.main_hash_head_bits is not None:
        if any(bit <= 0 for bit in args.main_hash_head_bits):
            parser.error("--main_hash_head_bits values must be positive")
    if args.union_hash_head_bits is not None:
        if any(bit <= 0 for bit in args.union_hash_head_bits):
            parser.error("--union_hash_head_bits values must be positive")
    if args.topology_hash_head_bits is not None:
        if any(bit <= 0 for bit in args.topology_hash_head_bits):
            parser.error("--topology_hash_head_bits values must be positive")
    if args.structure_degree_ratio_max < 1.0:
        parser.error("--structure_degree_ratio_max must be at least 1.0")
    if args.structure_homophily_gap_max is not None and args.structure_homophily_gap_max < 0:
        parser.error("--structure_homophily_gap_max must be non-negative")
    if args.topology_sketch_bits <= 0:
        parser.error("--topology_sketch_bits must be positive")
    if args.topology_sketch_radius < 0 or args.topology_sketch_radius > args.topology_sketch_bits:
        parser.error("--topology_sketch_radius must be between 0 and --topology_sketch_bits")
    if args.topology_degree_bucket_gap is not None and args.topology_degree_bucket_gap < 0:
        parser.error("--topology_degree_bucket_gap must be non-negative when provided")
    if args.topology_homophily_bins < 2:
        parser.error("--topology_homophily_bins must be at least 2")
    if args.topology_homophily_bucket_gap < 0:
        parser.error("--topology_homophily_bucket_gap must be non-negative")
    route_tags = list_retrieval_route_tags(
        args.hash_view,
        args.union_hash_views,
        include_topology=args.enable_topology_retrieval_route,
    )
    if args.route_score_weights is not None and len(args.route_score_weights) != len(route_tags):
        parser.error(f"--route_score_weights expects {len(route_tags)} values for the configured routes")
    if args.route_score_weights is not None and any(weight <= 0 for weight in args.route_score_weights):
        parser.error("--route_score_weights values must be positive")
    if args.union_route_weight <= 0:
        parser.error("--union_route_weight must be positive")
    if args.table_route_weight_decay <= 0:
        parser.error("--table_route_weight_decay must be positive")
    if args.route_accept_tau_offsets is not None and len(args.route_accept_tau_offsets) != len(route_tags):
        parser.error(f"--route_accept_tau_offsets expects {len(route_tags)} values for the configured routes")
    if args.route_min_accept_votes is not None and len(args.route_min_accept_votes) != len(route_tags):
        parser.error(f"--route_min_accept_votes expects {len(route_tags)} values for the configured routes")
    if args.route_min_support_hits is not None and len(args.route_min_support_hits) not in (1, len(route_tags)):
        parser.error(f"--route_min_support_hits expects 1 or {len(route_tags)} values for the configured routes")
    if args.union_accept_tau_bonus < 0:
        parser.error("--union_accept_tau_bonus must be non-negative")
    if args.route_min_accept_votes is not None and any(vote < 1 for vote in args.route_min_accept_votes):
        parser.error("--route_min_accept_votes values must be at least 1")
    if args.route_min_support_hits is not None and any(hit < 1 for hit in args.route_min_support_hits):
        parser.error("--route_min_support_hits values must be at least 1")
    if args.union_min_accept_votes < 1:
        parser.error("--union_min_accept_votes must be at least 1")
    if args.union_min_support_hits < 1:
        parser.error("--union_min_support_hits must be at least 1")
    if args.min_base_route_hits < 1:
        parser.error("--min_base_route_hits must be at least 1")
    if args.max_candidates_per_route < 1:
        parser.error("--max_candidates_per_route must be at least 1")
    if args.max_total_candidates < 1:
        parser.error("--max_total_candidates must be at least 1")
    if args.max_total_candidates < args.max_candidates_per_route:
        parser.error("--max_total_candidates must be >= --max_candidates_per_route")
    if args.max_structure_checks is not None and args.max_structure_checks < 1:
        parser.error("--max_structure_checks must be at least 1")
    if args.coarse_union_bits_max is not None and args.coarse_union_bits_max <= 0:
        parser.error("--coarse_union_bits_max must be positive")
    if args.exact_guard_low_bits < 0:
        parser.error("--exact_guard_low_bits must be non-negative")
    if args.exact_guard_min_bucket_size < 1:
        parser.error("--exact_guard_min_bucket_size must be at least 1")
    if args.exact_guard_large_bucket_size < args.exact_guard_min_bucket_size:
        parser.error("--exact_guard_large_bucket_size must be >= --exact_guard_min_bucket_size")
    if args.exact_guard_min_margin < 0:
        parser.error("--exact_guard_min_margin must be non-negative")
    if args.exact_guard_cosine_bonus < 0:
        parser.error("--exact_guard_cosine_bonus must be non-negative")
    if args.score_reuse_threshold < 0:
        parser.error("--score_reuse_threshold must be non-negative")
    if not (0 <= args.score_hub_threshold <= 15):
        parser.error("--score_hub_threshold must be in [0, 15]")
    if not (0 <= args.score_rare_threshold <= 15):
        parser.error("--score_rare_threshold must be in [0, 15]")
    if args.score_rarity_bits <= 0:
        parser.error("--score_rarity_bits must be positive")
    if args.score_rare_min_dist < 1:
        parser.error("--score_rare_min_dist must be positive")
    if args.score_rare_min_route_hits < 1 or args.score_rare_min_base_hits < 1:
        parser.error("score rare support thresholds must be positive")
    if args.score_pair_confidence_discount < 0:
        parser.error("--score_pair_confidence_discount must be non-negative")
    if args.score_pair_confidence_max_dist < 0:
        parser.error("--score_pair_confidence_max_dist must be non-negative")
    if args.score_pair_confidence_min_route_hits < 1 or args.score_pair_confidence_min_base_hits < 1:
        parser.error("score pair confidence support thresholds must be positive")
    if args.score_propagation_weight < 0 or args.score_graph_context_weight < 0 or args.score_low_unique_weight < 0:
        parser.error("score weights must be non-negative")
    if args.quant_int4_threshold < 0 or args.quant_int8_threshold < 0:
        parser.error("quant thresholds must be non-negative")
    if args.quant_int4_error <= 0 or args.quant_int8_error <= 0:
        parser.error("quant approximation errors must be positive")
    if args.quant_int4_bits < 2 or args.quant_int8_bits < 2:
        parser.error("quant fake bits must be at least 2")
    if args.quant_int4_bits > args.quant_int8_bits:
        parser.error("--quant_int4_bits should be <= --quant_int8_bits")
    if (
        args.quant_tser_propagation_weight < 0
        or args.quant_tser_graph_context_weight < 0
        or args.quant_tser_low_unique_weight < 0
    ):
        parser.error("quant TSER weights must be non-negative")
    if args.quant_error_bias < 0:
        parser.error("--quant_error_bias must be non-negative")
    if args.residual_rank <= 0:
        parser.error("--residual_rank must be positive")
    if args.residual_epochs < 0:
        parser.error("--residual_epochs must be non-negative")
    if args.residual_lr <= 0:
        parser.error("--residual_lr must be positive")
    if args.residual_weight_decay < 0:
        parser.error("--residual_weight_decay must be non-negative")
    if args.residual_l2 < 0:
        parser.error("--residual_l2 must be non-negative")
    if not args.residual_alpha_grid:
        parser.error("--residual_alpha_grid must contain at least one value")
    if any(alpha < 0 for alpha in args.residual_alpha_grid):
        parser.error("--residual_alpha_grid values must be non-negative")
    if args.residual_min_dist < 0:
        parser.error("--residual_min_dist must be non-negative")
    if args.residual_direct_threshold < -1:
        parser.error("--residual_direct_threshold must be >= -1")
    if args.residual_min_route_hits < 1 or args.residual_min_base_hits < 1:
        parser.error("--residual_min_route_hits/--residual_min_base_hits must be positive")
    if args.residual_hard_min_support_hits != -1 and args.residual_hard_min_support_hits < 1:
        parser.error("--residual_hard_min_support_hits must be -1 or positive")
    if args.residual_soft_min_support_hits != -1 and args.residual_soft_min_support_hits < 1:
        parser.error("--residual_soft_min_support_hits must be -1 or positive")
    if args.residual_hard_min_support_hits > 0 or args.residual_soft_min_support_hits > 0:
        if args.residual_hard_min_support_hits <= 0 or args.residual_soft_min_support_hits <= 0:
            parser.error("residual support split requires both hard and soft support thresholds")
        if args.residual_soft_min_support_hits >= args.residual_hard_min_support_hits:
            parser.error("--residual_soft_min_support_hits must be smaller than --residual_hard_min_support_hits")
    if args.residual_max_train_pairs < 0:
        parser.error("--residual_max_train_pairs must be non-negative")
    if args.internal_calib_samples < 0:
        parser.error("--internal_calib_samples must be >= 0")
    if not (0.0 <= args.internal_calib_high_ratio <= 1.0):
        parser.error("--internal_calib_high_ratio must be in [0, 1]")
    if not (0.0 <= args.internal_split_topk_ratio <= 1.0):
        parser.error("--internal_split_topk_ratio must be in [0, 1]")
    if not (args.internal_split_mass_target < 0.0 or 0.0 <= args.internal_split_mass_target <= 1.0):
        parser.error("--internal_split_mass_target must be negative or in [0, 1]")
    if args.internal_calib_bucket_count <= 0:
        parser.error("--internal_calib_bucket_count must be positive")
    if not (args.internal_calib_bucket_topk_ratio < 0.0 or 0.0 <= args.internal_calib_bucket_topk_ratio <= 1.0):
        parser.error("--internal_calib_bucket_topk_ratio must be negative or in [0, 1]")
    if not (0.0 <= args.internal_calib_uniform_ratio <= 1.0):
        parser.error("--internal_calib_uniform_ratio must be in [0, 1]")
    if args.internal_calib_high_a_bit <= 0 or args.internal_calib_low_a_bit <= 0:
        parser.error("--internal_calib_high_a_bit/--internal_calib_low_a_bit must be positive")
    if args.internal_calib_high_a_bit < args.internal_calib_low_a_bit:
        parser.error("--internal_calib_high_a_bit must be >= --internal_calib_low_a_bit")
    if args.internal_split_calibration_only and not args.internal_split_calibration:
        args.internal_split_calibration = True
    if args.real_quant_error_norm <= 0:
        parser.error("--real_quant_error_norm must be positive")
    if args.real_quant_int4_threshold < 0 or args.real_quant_int8_threshold < 0:
        parser.error("real quant thresholds must be non-negative")
    if not (0.0 <= args.real_quant_fp_ratio <= 1.0):
        parser.error("--real_quant_fp_ratio must be in [0, 1]")
    if not (0.0 <= args.real_quant_int8_ratio <= 1.0):
        parser.error("--real_quant_int8_ratio must be in [0, 1]")
    if args.real_quant_fp_ratio + args.real_quant_int8_ratio > 1.0:
        parser.error("--real_quant_fp_ratio + --real_quant_int8_ratio must be <= 1")
    if not args.graph_eager_token_lengths:
        parser.error("--graph_eager_token_lengths must contain at least one length")
    if any(length <= 0 for length in args.graph_eager_token_lengths):
        parser.error("--graph_eager_token_lengths must be positive")
    if args.graph_eager_full_length <= 0:
        parser.error("--graph_eager_full_length must be positive")
    if max(args.graph_eager_token_lengths) >= args.graph_eager_full_length:
        parser.error("--graph_eager_token_lengths must be smaller than --graph_eager_full_length")
    if args.graph_eager_cost_scale <= 0:
        parser.error("--graph_eager_cost_scale must be positive")
    if args.graph_eager_attn_weight < 0 or args.graph_eager_ffn_weight < 0:
        parser.error("--graph_eager_attn_weight and --graph_eager_ffn_weight must be non-negative")
    if args.graph_eager_attn_weight + args.graph_eager_ffn_weight <= 0:
        parser.error("--graph_eager_attn_weight + --graph_eager_ffn_weight must be positive")
    if not (0.0 <= args.graph_eager_full_ratio <= 1.0):
        parser.error("--graph_eager_full_ratio must be in [0, 1]")
    if not (0.0 <= args.graph_eager_mid_ratio <= 1.0):
        parser.error("--graph_eager_mid_ratio must be in [0, 1]")
    if args.graph_eager_full_ratio + args.graph_eager_mid_ratio > 1.0:
        parser.error("--graph_eager_full_ratio + --graph_eager_mid_ratio must be <= 1")
    if args.graph_eager_predictor_calib_samples <= 0:
        parser.error("--graph_eager_predictor_calib_samples must be positive")
    if args.graph_eager_predictor_ridge < 0:
        parser.error("--graph_eager_predictor_ridge must be non-negative")
    if not args.token_compaction_tags:
        parser.error("--token_compaction_tags must contain at least one tag")
    if args.token_compaction_names is not None and len(args.token_compaction_names) != len(args.token_compaction_tags):
        parser.error("--token_compaction_names must match --token_compaction_tags length")
    if args.token_compaction_length <= 0:
        parser.error("--token_compaction_length must be positive")
    if not args.ffn_gating_tags:
        parser.error("--ffn_gating_tags must contain at least one tag")
    if args.ffn_gating_names is not None and len(args.ffn_gating_names) != len(args.ffn_gating_tags):
        parser.error("--ffn_gating_names must match --ffn_gating_tags length")
    if len(args.ffn_gating_keep_ratios) != len(args.ffn_gating_tags):
        parser.error("--ffn_gating_keep_ratios must match --ffn_gating_tags length")
    if any(ratio <= 0.0 or ratio > 1.0 for ratio in args.ffn_gating_keep_ratios):
        parser.error("--ffn_gating_keep_ratios values must be in (0, 1]")
    if not args.ffn_gating_route_ratios:
        parser.error("--ffn_gating_route_ratios must contain at least one value")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in args.ffn_gating_route_ratios):
        parser.error("--ffn_gating_route_ratios values must be in [0, 1]")
    if args.ffn_gating_cost_scale <= 0:
        parser.error("--ffn_gating_cost_scale must be positive")
    if args.ffn_gating_attn_weight < 0 or args.ffn_gating_ffn_weight < 0:
        parser.error("--ffn_gating_attn_weight and --ffn_gating_ffn_weight must be non-negative")
    if args.ffn_gating_attn_weight + args.ffn_gating_ffn_weight <= 0:
        parser.error("--ffn_gating_attn_weight + --ffn_gating_ffn_weight must be positive")
    if not (0.0 < args.hierarchical_gated_keep_ratio <= 1.0):
        parser.error("--hierarchical_gated_keep_ratio must be in (0, 1]")
    if not (0.0 <= args.hierarchical_gated_route_ratio <= 1.0):
        parser.error("--hierarchical_gated_route_ratio must be in [0, 1]")
    if args.hierarchical_residual_cost < 0:
        parser.error("--hierarchical_residual_cost must be non-negative")
    if args.controller_seed is not None and args.controller_seed < 0:
        parser.error("--controller_seed must be >= 0")
    if args.hash_head_seed is not None and args.hash_head_seed < 0:
        parser.error("--hash_head_seed must be >= 0")
    if args.topology_sketch_seed is not None and args.topology_sketch_seed < 0:
        parser.error("--topology_sketch_seed must be >= 0")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run_adaptive_simulation(args)
