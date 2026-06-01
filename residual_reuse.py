import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import _compute_neighbor_mean


class LowRankResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank=32):
        super().__init__()
        self.supports_gate = False
        self.down = nn.Linear(int(input_dim), int(rank))
        self.up = nn.Linear(int(rank), int(output_dim))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward_with_gate(self, x):
        delta = self.up(F.gelu(self.down(x)))
        corr_gate = torch.ones(delta.size(0), 1, dtype=delta.dtype, device=delta.device)
        accept_gate = torch.ones(delta.size(0), 1, dtype=delta.dtype, device=delta.device)
        return delta, corr_gate, accept_gate

    def forward(self, x):
        delta, corr_gate, _accept_gate = self.forward_with_gate(x)
        return delta * corr_gate


class ResidualMLPAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        rank=32,
        hidden_dim=0,
        hidden_layers=2,
        dropout=0.1,
        accept_mode="shared",
    ):
        super().__init__()
        self.supports_gate = True
        self.accept_mode = str(accept_mode)
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        rank = int(rank)
        hidden_layers = max(1, int(hidden_layers))
        hidden_dim = int(hidden_dim) if int(hidden_dim) > 0 else max(128, rank * 4)

        self.input_norm = nn.LayerNorm(input_dim)
        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.latent = nn.Linear(dim, rank)
        self.out = nn.Linear(rank, output_dim)
        self.scale = nn.Linear(dim, 1)
        self.accept = None if self.accept_mode == "shared" else nn.Linear(dim, 1)

        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, -1.0)
        if self.accept is not None:
            nn.init.zeros_(self.accept.weight)
            nn.init.constant_(self.accept.bias, 1.0)

    def forward_with_gate(self, x):
        h = self.trunk(self.input_norm(x))
        z = F.gelu(self.latent(h))
        delta = self.out(z)
        corr_gain = torch.sigmoid(self.scale(h))
        if self.accept is None:
            accept_gain = corr_gain
        else:
            accept_gain = torch.sigmoid(self.accept(h))
        return delta, corr_gain, accept_gain

    def forward(self, x):
        delta, corr_gain, _accept_gain = self.forward_with_gate(x)
        return delta * corr_gain


class SupportAwareResidualAdapter:
    def __init__(self, global_adapter, adapters_by_support=None, bucket_mode="support"):
        self.global_adapter = global_adapter
        self.adapters_by_support = dict(adapters_by_support or {})
        self.bucket_mode = str(bucket_mode)


def _as_float_dict(mapping):
    return {int(key): float(value) for key, value in dict(mapping or {}).items()}


def compute_bucket_values_from_tensors(support_hits, best_dists, bucket_mode="support"):
    support_hits = support_hits.to(dtype=torch.long)
    best_dists = best_dists.to(dtype=torch.long).clamp(min=0, max=3)
    if str(bucket_mode) == "support":
        return support_hits
    if str(bucket_mode) == "support_dist":
        return support_hits * 10 + best_dists
    raise ValueError(f"Unknown residual bucket mode: {bucket_mode}")


def compute_bucket_values_from_trace(trace, node_indices, bucket_mode="support", device=None):
    node_indices = node_indices.to(dtype=torch.long, device=trace["winning_base_table_hit_counts"].device)
    support_hits = trace["winning_base_table_hit_counts"][node_indices]
    best_dists = trace["best_dists"][node_indices]
    bucket_values = compute_bucket_values_from_tensors(support_hits, best_dists, bucket_mode=bucket_mode)
    if device is not None:
        bucket_values = bucket_values.to(device=device)
    return bucket_values


def format_bucket_label(bucket_value, bucket_mode="support"):
    bucket_value = int(bucket_value)
    if str(bucket_mode) == "support":
        return f"{bucket_value}h"
    if str(bucket_mode) == "support_dist":
        support = bucket_value // 10
        dist = bucket_value % 10
        return f"{support}h_d{dist}"
    return str(bucket_value)


def _offline_bucket_rank(query_support, query_dist, cand_support, cand_dist, bucket_mode="support_dist"):
    query_support = int(query_support)
    query_dist = int(query_dist)
    cand_support = int(cand_support)
    cand_dist = int(cand_dist)
    bucket_mode = str(bucket_mode)

    if bucket_mode == "support":
        if cand_support == query_support:
            return 0
        if abs(cand_support - query_support) <= 1:
            return 1
        return None

    if bucket_mode == "support_dist":
        if cand_support == query_support and cand_dist == query_dist:
            return 0
        if cand_support == query_support and abs(cand_dist - query_dist) <= 1:
            return 1
        if abs(cand_support - query_support) <= 1 and abs(cand_dist - query_dist) <= 1:
            return 2
        return None

    raise ValueError(f"Unknown residual bucket mode: {bucket_mode}")


def compute_total_degree(edge_index, num_nodes, device):
    row, col = edge_index
    sym_nodes = torch.cat([row, col], dim=0)
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, sym_nodes, torch.ones(sym_nodes.numel(), dtype=torch.float32, device=device))
    return degree


def build_context_signature(verify_features, edge_index):
    neighbor_mean = _compute_neighbor_mean(verify_features, edge_index)
    return F.normalize(0.5 * verify_features + 0.5 * neighbor_mean, p=2, dim=1)


def build_residual_pair_inputs(
    verify_features,
    edge_index,
    trace,
    node_indices,
    risk_scores=None,
    source_ids=None,
    best_dists=None,
    route_hit_counts=None,
    base_route_hit_counts=None,
    winning_base_table_hit_counts=None,
    best_cosines=None,
):
    device = verify_features.device
    node_indices = node_indices.to(device=device, dtype=torch.long)
    if source_ids is None:
        source_ids = trace["source_ids"][node_indices]
    source_ids = source_ids.to(device=device, dtype=torch.long)

    context = build_context_signature(verify_features, edge_index)
    degree = compute_total_degree(edge_index, verify_features.size(0), device)

    query_verify = verify_features[node_indices]
    anchor_verify = verify_features[source_ids]
    query_context = context[node_indices]
    anchor_context = context[source_ids]
    cheap_delta = verify_features[node_indices] - verify_features[source_ids]
    context_delta = context[node_indices] - context[source_ids]

    if best_dists is None:
        best_dists = trace["best_dists"][node_indices]
    if route_hit_counts is None:
        route_hit_counts = trace["route_hit_counts"][node_indices]
    if base_route_hit_counts is None:
        base_route_hit_counts = trace["base_route_hit_counts"][node_indices]
    if winning_base_table_hit_counts is None:
        winning_base_table_hit_counts = trace["winning_base_table_hit_counts"][node_indices]
    if best_cosines is None:
        best_cosines = trace["best_cosines"][node_indices]

    dist = best_dists.to(device=device, dtype=torch.float32).clamp(min=0.0)
    route_hits = route_hit_counts.to(device=device, dtype=torch.float32).clamp(min=0.0)
    base_hits = base_route_hit_counts.to(device=device, dtype=torch.float32).clamp(min=0.0)
    base_table_hits = winning_base_table_hit_counts.to(device=device, dtype=torch.float32).clamp(min=0.0)
    best_cos = best_cosines.to(device=device, dtype=torch.float32)

    deg_v = degree[node_indices]
    deg_u = degree[source_ids]
    log_degree_ratio = torch.log1p(deg_v) - torch.log1p(deg_u)
    cheap_cos = F.cosine_similarity(verify_features[node_indices], verify_features[source_ids], dim=1)
    context_cos = F.cosine_similarity(context[node_indices], context[source_ids], dim=1)

    if risk_scores is not None and "sensitivity_q" in risk_scores:
        sensitivity = risk_scores["sensitivity_q"][node_indices].to(device=device, dtype=torch.float32) / 64.0
    else:
        sensitivity = torch.zeros_like(dist)

    scalars = torch.stack(
        [
            dist / 16.0,
            route_hits / 8.0,
            base_hits / 4.0,
            base_table_hits / 8.0,
            best_cos,
            cheap_cos,
            context_cos,
            log_degree_ratio,
            sensitivity,
        ],
        dim=1,
    )
    return torch.cat(
        [
            query_verify,
            anchor_verify,
            query_context,
            anchor_context,
            cheap_delta,
            context_delta,
            scalars,
        ],
        dim=1,
    )


def _build_stratified_buckets(trace, indices):
    if indices.numel() == 0:
        return {}
    support = trace["winning_base_table_hit_counts"][indices].to(dtype=torch.long, device="cpu")
    dist = trace["best_dists"][indices].to(dtype=torch.long, device="cpu").clamp(min=0)
    dist = torch.clamp(dist, max=3)
    buckets = {}
    for pos in range(indices.numel()):
        key = (int(support[pos].item()), int(dist[pos].item()))
        buckets.setdefault(key, []).append(int(indices[pos].item()))
    return buckets


def _stratified_subsample(trace, indices, max_pairs):
    if int(max_pairs) <= 0 or indices.numel() <= int(max_pairs):
        return indices

    buckets = _build_stratified_buckets(trace, indices)
    if not buckets:
        return indices

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    ordered_keys = sorted(buckets.keys())
    per_bucket_quota = max(1, int(max_pairs) // max(1, len(ordered_keys)))
    selected = []
    leftovers = []

    for key in ordered_keys:
        bucket = torch.as_tensor(buckets[key], dtype=torch.long)
        perm = torch.randperm(bucket.numel(), generator=generator)
        bucket = bucket[perm]
        take = min(bucket.numel(), per_bucket_quota)
        if take > 0:
            selected.append(bucket[:take])
        if take < bucket.numel():
            leftovers.append(bucket[take:])

    selected_count = int(sum(chunk.numel() for chunk in selected))
    remaining = int(max_pairs) - selected_count
    if remaining > 0 and leftovers:
        pool = torch.cat(leftovers, dim=0)
        perm = torch.randperm(pool.numel(), generator=generator)
        pool = pool[perm]
        selected.append(pool[:remaining])

    sampled = torch.cat(selected, dim=0)
    return sampled.to(device=indices.device, dtype=torch.long)


def _build_residual_adapter(
    input_dim,
    output_dim,
    rank,
    adapter_type="low_rank",
    hidden_dim=0,
    hidden_layers=2,
    dropout=0.1,
    accept_mode="shared",
):
    adapter_type = str(adapter_type).lower()
    if adapter_type == "low_rank":
        return LowRankResidualAdapter(input_dim, output_dim, rank=rank)
    if adapter_type == "mlp":
        return ResidualMLPAdapter(
            input_dim,
            output_dim,
            rank=rank,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            accept_mode=accept_mode,
        )
    raise ValueError(f"Unknown residual adapter type: {adapter_type}")


def _adapter_forward(adapter, x):
    if hasattr(adapter, "forward_with_gate"):
        outputs = adapter.forward_with_gate(x)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            raw_delta, correction_gate, accept_gate = outputs
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            raw_delta, correction_gate = outputs
            accept_gate = correction_gate
        else:
            raise ValueError("forward_with_gate must return (delta, corr_gate) or (delta, corr_gate, accept_gate)")
    else:
        raw_delta = adapter(x)
        correction_gate = torch.ones(raw_delta.size(0), 1, dtype=raw_delta.dtype, device=raw_delta.device)
        accept_gate = torch.ones(raw_delta.size(0), 1, dtype=raw_delta.dtype, device=raw_delta.device)
    accept_gate = _apply_accept_calibration(adapter, accept_gate)
    if not bool(getattr(adapter, "supports_gate", False)):
        correction_gate = None
        gated_residual = raw_delta
    else:
        gated_residual = raw_delta * correction_gate
    return gated_residual, raw_delta, correction_gate, accept_gate


def _apply_accept_calibration(adapter, accept_gate):
    if accept_gate is None:
        return None
    scale = getattr(adapter, "accept_calibration_scale", None)
    bias = getattr(adapter, "accept_calibration_bias", None)
    if scale is None or bias is None:
        return accept_gate
    logits = torch.logit(accept_gate.clamp(min=1e-4, max=1.0 - 1e-4))
    return torch.sigmoid(logits * float(scale) + float(bias))


def _fit_accept_calibration(positive_accept_gate, positive_targets=None, negative_accept_gate=None):
    probs = []
    labels = []

    if positive_accept_gate is not None and int(positive_accept_gate.numel()) > 0:
        pos_prob = positive_accept_gate.detach().view(-1, 1).to(dtype=torch.float32)
        if positive_targets is None:
            pos_target = torch.ones_like(pos_prob)
        else:
            pos_target = positive_targets.detach().view(-1, 1).to(device=pos_prob.device, dtype=pos_prob.dtype)
        probs.append(pos_prob)
        labels.append(pos_target)

    if negative_accept_gate is not None and int(negative_accept_gate.numel()) > 0:
        neg_prob = negative_accept_gate.detach().view(-1, 1).to(dtype=torch.float32)
        neg_target = torch.zeros_like(neg_prob)
        probs.append(neg_prob)
        labels.append(neg_target)

    if not probs:
        return None

    prob_tensor = torch.cat(probs, dim=0).clamp(min=1e-4, max=1.0 - 1e-4)
    label_tensor = torch.cat(labels, dim=0)
    positives = int((label_tensor >= 0.5).sum().item())
    negatives = int((label_tensor < 0.5).sum().item())
    if positives == 0 or negatives == 0:
        return None

    logit_tensor = torch.logit(prob_tensor)
    scale = torch.nn.Parameter(torch.ones(1, device=logit_tensor.device, dtype=logit_tensor.dtype))
    bias = torch.nn.Parameter(torch.zeros(1, device=logit_tensor.device, dtype=logit_tensor.dtype))
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.5, max_iter=64, line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        calibrated_logits = logit_tensor * scale + bias
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, label_tensor)
        loss.backward()
        return loss

    try:
        optimizer.step(_closure)
    except RuntimeError:
        return None

    with torch.no_grad():
        calibrated = torch.sigmoid(logit_tensor * scale + bias)
        loss = F.binary_cross_entropy(calibrated, label_tensor)
    return {
        "scale": float(scale.detach().item()),
        "bias": float(bias.detach().item()),
        "samples": int(label_tensor.numel()),
        "positives": positives,
        "negatives": negatives,
        "loss": float(loss.detach().item()),
    }


def _extract_accept_calibration(adapter):
    scale = getattr(adapter, "accept_calibration_scale", None)
    bias = getattr(adapter, "accept_calibration_bias", None)
    if scale is None or bias is None:
        return None
    return {
        "scale": float(scale),
        "bias": float(bias),
        "samples": int(getattr(adapter, "accept_calibration_samples", 0)),
        "positives": int(getattr(adapter, "accept_calibration_positives", 0)),
        "negatives": int(getattr(adapter, "accept_calibration_negatives", 0)),
        "loss": float(getattr(adapter, "accept_calibration_loss", 0.0)),
    }


def _compute_residual_loss(
    residual,
    anchors,
    targets,
    cosine_weight,
    mse_weight,
    delta_weight,
    residual_l2,
    raw_delta=None,
    correction_gate=None,
    accept_gate=None,
    gate_loss_weight=0.0,
    accept_loss_weight=0.0,
    gate_error_scale=0.25,
    gate_error_max=0.45,
    gate_sparsity_weight=0.0,
    accept_targets=None,
):
    pred_raw = anchors + residual
    pred_norm = F.normalize(pred_raw, p=2, dim=1)
    target_norm = F.normalize(targets, p=2, dim=1)
    target_delta = targets - anchors

    loss = pred_raw.new_tensor(0.0)
    if float(cosine_weight) > 0.0:
        loss = loss + float(cosine_weight) * (1.0 - F.cosine_similarity(pred_norm, target_norm, dim=1)).mean()
    if float(mse_weight) > 0.0:
        loss = loss + float(mse_weight) * F.smooth_l1_loss(pred_raw, targets)
    if float(delta_weight) > 0.0:
        loss = loss + float(delta_weight) * F.smooth_l1_loss(residual, target_delta)
    if float(residual_l2) > 0.0:
        loss = loss + float(residual_l2) * residual.pow(2).mean()
    if correction_gate is not None:
        anchor_norm = F.normalize(anchors, p=2, dim=1)
        anchor_error = (1.0 - F.cosine_similarity(anchor_norm, target_norm, dim=1)).clamp(min=0.0)
        error_scale = max(1e-6, float(gate_error_scale))
        error_max = max(error_scale + 1e-6, float(gate_error_max))
        rise = torch.clamp(anchor_error / error_scale, min=0.0, max=1.0)
        fall = torch.clamp((error_max - anchor_error) / max(1e-6, error_max - error_scale), min=0.0, max=1.0)
        gate_target = (rise * fall).unsqueeze(1)
        if float(gate_loss_weight) > 0.0:
            loss = loss + float(gate_loss_weight) * F.smooth_l1_loss(correction_gate, gate_target)
        if float(gate_sparsity_weight) > 0.0:
            low_error_weight = (1.0 - gate_target).detach()
            loss = loss + float(gate_sparsity_weight) * (correction_gate * low_error_weight).mean()
            if raw_delta is not None:
                delta_norm = raw_delta.pow(2).mean(dim=1, keepdim=True)
                loss = loss + float(gate_sparsity_weight) * 0.5 * (delta_norm * low_error_weight).mean()
    if accept_gate is not None and float(accept_loss_weight) > 0.0:
        if accept_targets is None:
            accept_target = torch.ones_like(accept_gate)
        else:
            accept_target = accept_targets.to(device=accept_gate.device, dtype=accept_gate.dtype).view_as(accept_gate)
        accept_prob = accept_gate.clamp(min=1e-4, max=1.0 - 1e-4)
        loss = loss + float(accept_loss_weight) * F.binary_cross_entropy(accept_prob, accept_target)
    return loss


def _fit_residual_adapter(
    x_train,
    anchors,
    targets,
    rank,
    epochs,
    lr,
    weight_decay,
    residual_l2,
    adapter_type="low_rank",
    hidden_dim=0,
    hidden_layers=2,
    dropout=0.1,
    accept_mode="shared",
    cosine_weight=1.0,
    mse_weight=0.25,
    delta_weight=0.5,
    gate_loss_weight=0.0,
    accept_loss_weight=0.0,
    gate_error_scale=0.25,
    gate_error_max=0.45,
    gate_sparsity_weight=0.0,
    x_neg=None,
    negative_gate_weight=1.0,
    accept_targets=None,
):
    device = anchors.device
    adapter = _build_residual_adapter(
        x_train.size(1),
        targets.size(1),
        rank=rank,
        adapter_type=adapter_type,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        accept_mode=accept_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    last_loss = 0.0
    correction_gate_mean = 1.0
    accept_gate_mean = 1.0
    negative_accept_gate_eval = None
    for _epoch in range(int(epochs)):
        adapter.train()
        optimizer.zero_grad()
        residual, raw_delta, correction_gate, accept_gate = _adapter_forward(adapter, x_train)
        loss = _compute_residual_loss(
            residual,
            anchors,
            targets,
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            delta_weight=delta_weight,
            residual_l2=residual_l2,
            raw_delta=raw_delta,
            correction_gate=correction_gate,
            accept_gate=accept_gate,
            gate_loss_weight=gate_loss_weight,
            accept_loss_weight=accept_loss_weight,
            gate_error_scale=gate_error_scale,
            gate_error_max=gate_error_max,
            gate_sparsity_weight=gate_sparsity_weight,
            accept_targets=accept_targets,
        )
        if x_neg is not None and int(x_neg.size(0)) > 0:
            neg_residual, neg_raw_delta, neg_correction_gate, neg_accept_gate = _adapter_forward(adapter, x_neg)
            neg_loss = neg_residual.new_tensor(0.0)
            if neg_correction_gate is not None:
                zeros = torch.zeros_like(neg_correction_gate)
                if float(gate_loss_weight) > 0.0:
                    neg_loss = neg_loss + float(gate_loss_weight) * F.smooth_l1_loss(neg_correction_gate, zeros)
                if float(gate_sparsity_weight) > 0.0:
                    neg_loss = neg_loss + float(gate_sparsity_weight) * neg_correction_gate.mean()
            if neg_accept_gate is not None and float(accept_loss_weight) > 0.0:
                zeros = torch.zeros_like(neg_accept_gate)
                neg_prob = neg_accept_gate.clamp(min=1e-4, max=1.0 - 1e-4)
                neg_loss = neg_loss + float(accept_loss_weight) * F.binary_cross_entropy(neg_prob, zeros)
            if float(gate_sparsity_weight) > 0.0:
                neg_loss = neg_loss + float(gate_sparsity_weight) * neg_raw_delta.pow(2).mean()
            loss = loss + float(negative_gate_weight) * neg_loss
            negative_accept_gate_eval = neg_accept_gate.detach()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
        correction_gate_mean = 1.0 if correction_gate is None else float(correction_gate.detach().mean().item())
        accept_gate_mean = 1.0 if accept_gate is None else float(accept_gate.detach().mean().item())
    accept_calibration = _fit_accept_calibration(
        accept_gate,
        positive_targets=accept_targets,
        negative_accept_gate=negative_accept_gate_eval,
    )
    if accept_calibration is not None:
        adapter.accept_calibration_scale = float(accept_calibration["scale"])
        adapter.accept_calibration_bias = float(accept_calibration["bias"])
        adapter.accept_calibration_samples = int(accept_calibration["samples"])
        adapter.accept_calibration_positives = int(accept_calibration["positives"])
        adapter.accept_calibration_negatives = int(accept_calibration["negatives"])
        adapter.accept_calibration_loss = float(accept_calibration["loss"])
        with torch.no_grad():
            accept_gate_mean = float(_apply_accept_calibration(adapter, accept_gate).mean().item())
    return adapter, last_loss, correction_gate_mean, accept_gate_mean


def _build_class_accept_targets(data, pair_tensors, split):
    if pair_tensors is None or pair_tensors["node_indices"].numel() == 0:
        return None, 0, 0
    if not hasattr(data, "y") or data.y is None:
        return None, 0, 0

    device = pair_tensors["node_indices"].device
    labels = data.y.to(device=device)
    if labels.dim() > 1:
        labels = labels.argmax(dim=-1)
    labels = labels.view(-1)

    node_indices = pair_tensors["node_indices"].to(device=device, dtype=torch.long)
    source_ids = pair_tensors["source_ids"].to(device=device, dtype=torch.long)
    source_ok = (source_ids >= 0) & (source_ids < labels.numel())

    known_mask = _split_mask_for_train_pairs(data, split, device)
    safe_sources = source_ids.clamp(min=0, max=max(0, labels.numel() - 1))
    label_known = source_ok & known_mask[node_indices] & known_mask[safe_sources]
    if not bool(label_known.any().item()):
        return torch.ones(node_indices.numel(), 1, dtype=torch.float32, device=device), 0, 0

    targets = torch.ones(node_indices.numel(), 1, dtype=torch.float32, device=device)
    same_label = labels[node_indices[label_known]] == labels[source_ids[label_known]]
    targets[label_known, 0] = same_label.to(dtype=torch.float32)
    labelled = int(label_known.sum().item())
    positives = int(same_label.sum().item())
    return targets, labelled, positives


def _build_classifier_accept_targets(
    model,
    data,
    pair_tensors,
    target_embeddings,
    reference_logits,
    mode="both",
    scope="node",
    max_kl=0.2,
    candidate_embeddings=None,
):
    if (
        model is None
        or reference_logits is None
        or pair_tensors is None
        or pair_tensors["node_indices"].numel() == 0
    ):
        return None, 0, 0, 0.0

    device = target_embeddings.device
    node_indices = pair_tensors["node_indices"].to(device=device, dtype=torch.long)
    source_ids = pair_tensors["source_ids"].to(device=device, dtype=torch.long)
    reference_logits = reference_logits.to(device=device, dtype=torch.float32)
    ref_pred = reference_logits.argmax(dim=1)
    ref_prob = F.softmax(reference_logits, dim=1)
    known_mask = (data.train_mask | data.val_mask).to(device=device, dtype=torch.bool)
    local_nodes = None
    if str(scope) == "local_trainval":
        local_sets = [set([idx]) for idx in range(int(target_embeddings.size(0)))]
        edge_cpu = data.edge_index.detach().cpu()
        for u, v in edge_cpu.t().tolist():
            local_sets[int(u)].add(int(v))
            local_sets[int(v)].add(int(u))
        local_nodes = [
            torch.as_tensor(
                sorted(n for n in nodes if bool(known_mask[int(n)].item())),
                dtype=torch.long,
                device=device,
            )
            for nodes in local_sets
        ]

    mode = str(mode)
    max_kl = float(max_kl)
    targets = torch.zeros(node_indices.numel(), 1, dtype=torch.float32, device=device)
    kl_values = []
    was_training = bool(model.training)
    model.eval()

    with torch.no_grad():
        for pos in range(int(node_indices.numel())):
            node_idx = int(node_indices[pos].item())
            src = int(source_ids[pos].item())
            if src < 0 or src >= int(target_embeddings.size(0)):
                continue

            candidate_raw = target_embeddings.clone()
            if candidate_embeddings is None:
                candidate_raw[node_idx] = target_embeddings[src]
            else:
                candidate_raw[node_idx] = candidate_embeddings[pos].to(
                    device=device,
                    dtype=target_embeddings.dtype,
                )
            candidate_emb = model.encoder(candidate_raw)
            candidate_logits = model.forward_gnn_only(
                candidate_emb,
                data.edge_index,
                data.edge_type,
                data.edge_attr,
            )
            eval_nodes = local_nodes[node_idx] if local_nodes is not None else node_indices[pos : pos + 1]
            if eval_nodes.numel() == 0:
                eval_nodes = node_indices[pos : pos + 1]
            node_logits = candidate_logits[eval_nodes]
            pred_ok = bool((node_logits.argmax(dim=1) == ref_pred[eval_nodes]).all().item())
            kl_value = F.kl_div(
                F.log_softmax(node_logits, dim=1),
                ref_prob[eval_nodes],
                reduction="batchmean",
            )
            kl_scalar = float(kl_value.item())
            kl_values.append(kl_scalar)
            kl_ok = kl_scalar <= max_kl
            if mode == "pred":
                accept_ok = pred_ok
            elif mode == "kl":
                accept_ok = kl_ok
            else:
                accept_ok = pred_ok and kl_ok
            targets[pos, 0] = 1.0 if accept_ok else 0.0

    if was_training:
        model.train()

    positives = int(targets.sum().item())
    mean_kl = float(sum(kl_values) / max(1, len(kl_values)))
    return targets, int(node_indices.numel()), positives, mean_kl


def _split_mask_for_train_pairs(data, split, device):
    if split == "train":
        return data.train_mask.to(device=device, dtype=torch.bool)
    if split == "train_val":
        return (data.train_mask | data.val_mask).to(device=device, dtype=torch.bool)
    if split == "all_hits":
        return torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    raise ValueError(f"Unknown residual train split: {split}")


def _build_training_pair_tensors(trace, train_nodes, device):
    train_nodes = train_nodes.to(device=device, dtype=torch.long)
    return {
        "node_indices": train_nodes,
        "source_ids": trace["source_ids"][train_nodes].to(device=device, dtype=torch.long),
        "best_dists": trace["best_dists"][train_nodes].to(device=device, dtype=torch.long),
        "route_hit_counts": trace["route_hit_counts"][train_nodes].to(device=device, dtype=torch.long),
        "base_route_hit_counts": trace["base_route_hit_counts"][train_nodes].to(device=device, dtype=torch.long),
        "winning_base_table_hit_counts": trace["winning_base_table_hit_counts"][train_nodes].to(
            device=device, dtype=torch.long
        ),
        "best_cosines": trace["best_cosines"][train_nodes].to(device=device, dtype=torch.float32),
    }


def _append_training_pairs(pair_tensors, extra_pairs):
    if extra_pairs is None or extra_pairs["node_indices"].numel() == 0:
        return pair_tensors
    merged = {}
    for key in pair_tensors.keys():
        merged[key] = torch.cat([pair_tensors[key], extra_pairs[key].to(device=pair_tensors[key].device)], dim=0)
    return merged


def _filter_positive_pairs_by_error(pair_tensors, target_embeddings, positive_error_max):
    if pair_tensors is None:
        return None, 0, 0
    if float(positive_error_max) < 0.0:
        count = int(pair_tensors["node_indices"].numel())
        return pair_tensors, count, count

    node_indices = pair_tensors["node_indices"]
    source_ids = pair_tensors["source_ids"]
    if node_indices.numel() == 0:
        return pair_tensors, 0, 0

    query_targets = target_embeddings[node_indices]
    anchor_targets = target_embeddings[source_ids]
    errors = (1.0 - F.cosine_similarity(query_targets, anchor_targets, dim=1)).clamp(min=0.0)
    keep_mask = errors <= float(positive_error_max)
    kept = int(keep_mask.sum().item())
    total = int(node_indices.numel())
    filtered = {key: value[keep_mask] for key, value in pair_tensors.items()}
    return filtered, kept, total


def _candidate_ref_identity(ref):
    if isinstance(ref, dict):
        return int(ref["route_idx"]), int(ref["hash"])
    if isinstance(ref, (tuple, list)) and len(ref) >= 2:
        return int(ref[0]), int(ref[1])
    raise TypeError(f"Unsupported candidate ref type: {type(ref)!r}")


def _harvest_extra_anchor_pairs(
    controller,
    hash_route_features,
    verify_features,
    target_embeddings,
    trace,
    data,
    train_nodes,
    split,
    min_dist,
    extra_anchors_per_node,
    extra_query_nodes,
    max_extra_pairs,
    positive_error_max,
    bucket_mode="support_dist",
):
    if (
        controller is None
        or hash_route_features is None
        or int(extra_anchors_per_node) <= 0
        or int(max_extra_pairs) <= 0
        or train_nodes.numel() == 0
    ):
        return None, {
            "query_nodes": 0,
            "candidate_refs": 0,
            "scored": 0,
            "reject_same_source": 0,
            "reject_min_dist": 0,
            "reject_support": 0,
            "reject_error": 0,
            "accepted": 0,
        }

    device = verify_features.device
    allowed_mask = _split_mask_for_train_pairs(data, split, device)
    allowed_ids = set(int(idx) for idx in allowed_mask.nonzero(as_tuple=False).view(-1).detach().cpu().tolist())
    if not allowed_ids:
        return None, {
            "query_nodes": 0,
            "candidate_refs": 0,
            "scored": 0,
            "reject_same_source": 0,
            "reject_min_dist": 0,
            "reject_support": 0,
            "reject_error": 0,
            "accepted": 0,
        }

    train_node_set = set(int(idx) for idx in train_nodes.detach().cpu().tolist())
    candidate_support_floor = min(
        int(value) for value in trace["winning_base_table_hit_counts"][train_nodes].detach().cpu().tolist()
    )
    query_nodes = list(train_node_set)
    if int(extra_query_nodes) > 0:
        extra_candidates = allowed_mask.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        extra_candidates = [int(idx) for idx in extra_candidates if int(idx) not in train_node_set]
        if extra_candidates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0)
            perm = torch.randperm(len(extra_candidates), generator=generator).tolist()
            sample_count = min(int(extra_query_nodes), len(extra_candidates))
            query_nodes.extend(extra_candidates[idx] for idx in perm[:sample_count])

    hash_feature_routes = controller._normalize_hash_feature_routes(hash_route_features)
    all_hashes = [
        controller._compute_route_fingerprint(route_features, route_idx)
        for route_idx, route_features in enumerate(hash_feature_routes)
    ]
    target_embeddings_cpu = target_embeddings.detach().cpu()

    extra_node_indices = []
    extra_source_ids = []
    extra_best_dists = []
    extra_route_hits = []
    extra_base_hits = []
    extra_support_hits = []
    extra_best_cos = []
    stats = {
        "query_nodes": int(len(query_nodes)),
        "candidate_refs": 0,
        "scored": 0,
        "reject_same_source": 0,
        "reject_min_dist": 0,
        "reject_support": 0,
        "reject_error": 0,
        "accepted_exact_bucket": 0,
        "accepted_relaxed_bucket": 0,
        "accepted": 0,
    }

    for node_idx in query_nodes:
        if len(extra_node_indices) >= int(max_extra_pairs):
            break
        query_hashes = [route_hashes[int(node_idx)] for route_hashes in all_hashes]
        exact_refs = controller._find_exact_candidate_refs(query_hashes)
        fuzzy_refs = controller._collect_union_candidate_refs(query_hashes, int(controller.node_policies[int(node_idx)].item()))
        seen_pairs = set()
        candidate_refs = []
        for ref in exact_refs + fuzzy_refs:
            key = _candidate_ref_identity(ref)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidate_refs.append(ref)
        stats["candidate_refs"] += int(len(candidate_refs))
        scored = controller._collect_scored_candidates(
            candidate_refs,
            verify_features[int(node_idx)],
            exclude_node_id=int(node_idx),
            allowed_node_ids=allowed_ids,
        )
        stats["scored"] += int(len(scored))
        primary_source = int(trace["source_ids"][int(node_idx)].item())
        seen_sources = {primary_source} if primary_source >= 0 else set()
        query_target = target_embeddings_cpu[int(node_idx)]
        query_support = int(trace["winning_base_table_hit_counts"][int(node_idx)].item())
        query_dist = int(trace["best_dists"][int(node_idx)].item())
        taken = 0
        ranked_items = {0: [], 1: [], 2: []}
        for item in scored:
            src = int(item["entry"]["node_id"])
            support_hits = int(item.get("winning_base_table_hit_count", 1))
            dist = int(item.get("dist", 0))
            if src in seen_sources:
                stats["reject_same_source"] += 1
                continue
            if float(dist) < float(min_dist):
                stats["reject_min_dist"] += 1
                continue
            if support_hits < candidate_support_floor:
                stats["reject_support"] += 1
                continue
            bucket_rank = _offline_bucket_rank(
                query_support,
                query_dist,
                support_hits,
                dist,
                bucket_mode=bucket_mode,
            )
            if bucket_rank is None:
                stats["reject_support"] += 1
                continue
            ranked_items[bucket_rank].append((item, src, support_hits, dist))

        for bucket_rank in (0, 1, 2):
            if len(extra_node_indices) >= int(max_extra_pairs) or taken >= int(extra_anchors_per_node):
                break
            for item, src, support_hits, dist in ranked_items[bucket_rank]:
                if len(extra_node_indices) >= int(max_extra_pairs) or taken >= int(extra_anchors_per_node):
                    break
                if float(positive_error_max) >= 0.0:
                    anchor_error = float(
                        (
                            1.0
                            - F.cosine_similarity(
                                query_target.unsqueeze(0),
                                target_embeddings_cpu[src].unsqueeze(0),
                                dim=1,
                            )
                        ).item()
                    )
                    if anchor_error > float(positive_error_max):
                        stats["reject_error"] += 1
                        continue
                seen_sources.add(src)
                extra_node_indices.append(int(node_idx))
                extra_source_ids.append(src)
                extra_best_dists.append(dist)
                extra_route_hits.append(int(item.get("route_hit_count", 1)))
                extra_base_hits.append(int(item.get("base_route_hit_count", 1)))
                extra_support_hits.append(support_hits)
                extra_best_cos.append(float(item.get("cos", 0.0)))
                taken += 1
                stats["accepted"] += 1
                if bucket_rank == 0:
                    stats["accepted_exact_bucket"] += 1
                else:
                    stats["accepted_relaxed_bucket"] += 1

    if not extra_node_indices:
        return None, stats

    return {
        "node_indices": torch.as_tensor(extra_node_indices, dtype=torch.long, device=device),
        "source_ids": torch.as_tensor(extra_source_ids, dtype=torch.long, device=device),
        "best_dists": torch.as_tensor(extra_best_dists, dtype=torch.long, device=device),
        "route_hit_counts": torch.as_tensor(extra_route_hits, dtype=torch.long, device=device),
        "base_route_hit_counts": torch.as_tensor(extra_base_hits, dtype=torch.long, device=device),
        "winning_base_table_hit_counts": torch.as_tensor(extra_support_hits, dtype=torch.long, device=device),
        "best_cosines": torch.as_tensor(extra_best_cos, dtype=torch.float32, device=device),
    }, stats


def _harvest_negative_anchor_pairs(
    controller,
    hash_route_features,
    verify_features,
    target_embeddings,
    trace,
    data,
    train_nodes,
    split,
    extra_anchors_per_node,
    extra_query_nodes,
    max_extra_pairs,
    negative_error_min,
    bucket_mode="support_dist",
):
    if (
        controller is None
        or hash_route_features is None
        or int(extra_anchors_per_node) <= 0
        or int(max_extra_pairs) <= 0
        or train_nodes.numel() == 0
    ):
        return None, {
            "query_nodes": 0,
            "candidate_refs": 0,
            "scored": 0,
            "reject_same_source": 0,
            "reject_support": 0,
            "reject_error": 0,
            "accepted": 0,
        }

    device = verify_features.device
    allowed_mask = _split_mask_for_train_pairs(data, split, device)
    allowed_ids = set(int(idx) for idx in allowed_mask.nonzero(as_tuple=False).view(-1).detach().cpu().tolist())
    if not allowed_ids:
        return None, {
            "query_nodes": 0,
            "candidate_refs": 0,
            "scored": 0,
            "reject_same_source": 0,
            "reject_support": 0,
            "reject_error": 0,
            "accepted": 0,
        }

    train_node_set = set(int(idx) for idx in train_nodes.detach().cpu().tolist())
    candidate_support_floor = min(
        int(value) for value in trace["winning_base_table_hit_counts"][train_nodes].detach().cpu().tolist()
    )
    query_nodes = list(train_node_set)
    if int(extra_query_nodes) > 0:
        extra_candidates = allowed_mask.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        extra_candidates = [int(idx) for idx in extra_candidates if int(idx) not in train_node_set]
        if extra_candidates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1)
            perm = torch.randperm(len(extra_candidates), generator=generator).tolist()
            sample_count = min(int(extra_query_nodes), len(extra_candidates))
            query_nodes.extend(extra_candidates[idx] for idx in perm[:sample_count])

    hash_feature_routes = controller._normalize_hash_feature_routes(hash_route_features)
    all_hashes = [
        controller._compute_route_fingerprint(route_features, route_idx)
        for route_idx, route_features in enumerate(hash_feature_routes)
    ]

    target_embeddings_cpu = target_embeddings.detach().cpu()
    negative_node_indices = []
    negative_source_ids = []
    negative_best_dists = []
    negative_route_hits = []
    negative_base_hits = []
    negative_support_hits = []
    negative_best_cos = []
    stats = {
        "query_nodes": int(len(query_nodes)),
        "candidate_refs": 0,
        "scored": 0,
        "reject_same_source": 0,
        "reject_support": 0,
        "reject_error": 0,
        "accepted_exact_bucket": 0,
        "accepted_relaxed_bucket": 0,
        "accepted": 0,
    }

    for node_idx in query_nodes:
        if len(negative_node_indices) >= int(max_extra_pairs):
            break
        query_hashes = [route_hashes[int(node_idx)] for route_hashes in all_hashes]
        exact_refs = controller._find_exact_candidate_refs(query_hashes)
        fuzzy_refs = controller._collect_union_candidate_refs(
            query_hashes, int(controller.node_policies[int(node_idx)].item())
        )
        seen_pairs = set()
        candidate_refs = []
        for ref in exact_refs + fuzzy_refs:
            key = _candidate_ref_identity(ref)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidate_refs.append(ref)
        stats["candidate_refs"] += int(len(candidate_refs))

        scored = controller._collect_scored_candidates(
            candidate_refs,
            verify_features[int(node_idx)],
            exclude_node_id=int(node_idx),
            allowed_node_ids=allowed_ids,
        )
        stats["scored"] += int(len(scored))

        primary_source = int(trace["source_ids"][int(node_idx)].item())
        seen_sources = {primary_source} if primary_source >= 0 else set()
        query_target = target_embeddings_cpu[int(node_idx)]
        query_support = int(trace["winning_base_table_hit_counts"][int(node_idx)].item())
        query_dist = int(trace["best_dists"][int(node_idx)].item())
        taken = 0
        ranked_items = {0: [], 1: [], 2: []}
        for item in reversed(scored):
            src = int(item["entry"]["node_id"])
            support_hits = int(item.get("winning_base_table_hit_count", 1))
            dist = int(item.get("dist", 0))
            if src in seen_sources:
                stats["reject_same_source"] += 1
                continue
            if support_hits < candidate_support_floor:
                stats["reject_support"] += 1
                continue
            bucket_rank = _offline_bucket_rank(
                query_support,
                query_dist,
                support_hits,
                dist,
                bucket_mode=bucket_mode,
            )
            if bucket_rank is None:
                stats["reject_support"] += 1
                continue
            ranked_items[bucket_rank].append((item, src, support_hits, dist))

        for bucket_rank in (0, 1, 2):
            if len(negative_node_indices) >= int(max_extra_pairs) or taken >= int(extra_anchors_per_node):
                break
            for item, src, support_hits, dist in ranked_items[bucket_rank]:
                if len(negative_node_indices) >= int(max_extra_pairs) or taken >= int(extra_anchors_per_node):
                    break
                anchor_error = float(
                    (
                        1.0
                        - F.cosine_similarity(
                            query_target.unsqueeze(0),
                            target_embeddings_cpu[src].unsqueeze(0),
                            dim=1,
                        )
                    ).item()
                )
                if anchor_error < float(negative_error_min):
                    stats["reject_error"] += 1
                    continue
                seen_sources.add(src)
                negative_node_indices.append(int(node_idx))
                negative_source_ids.append(src)
                negative_best_dists.append(dist)
                negative_route_hits.append(int(item.get("route_hit_count", 1)))
                negative_base_hits.append(int(item.get("base_route_hit_count", 1)))
                negative_support_hits.append(support_hits)
                negative_best_cos.append(float(item.get("cos", 0.0)))
                taken += 1
                stats["accepted"] += 1
                if bucket_rank == 0:
                    stats["accepted_exact_bucket"] += 1
                else:
                    stats["accepted_relaxed_bucket"] += 1

    if not negative_node_indices:
        return None, stats

    return {
        "node_indices": torch.as_tensor(negative_node_indices, dtype=torch.long, device=device),
        "source_ids": torch.as_tensor(negative_source_ids, dtype=torch.long, device=device),
        "best_dists": torch.as_tensor(negative_best_dists, dtype=torch.long, device=device),
        "route_hit_counts": torch.as_tensor(negative_route_hits, dtype=torch.long, device=device),
        "base_route_hit_counts": torch.as_tensor(negative_base_hits, dtype=torch.long, device=device),
        "winning_base_table_hit_counts": torch.as_tensor(negative_support_hits, dtype=torch.long, device=device),
        "best_cosines": torch.as_tensor(negative_best_cos, dtype=torch.float32, device=device),
    }, stats


def select_residual_train_nodes(
    trace,
    data,
    split="train_val",
    max_pairs=4096,
    correction_mask=None,
    min_dist=0.0,
):
    hit_mask = trace["hit_mask"]
    source_ok = trace["source_ids"] >= 0
    mask = hit_mask & source_ok
    if correction_mask is not None:
        mask = mask & correction_mask.to(device=mask.device, dtype=torch.bool)
    if float(min_dist) > 0.0:
        mask = mask & (trace["best_dists"].to(device=mask.device, dtype=torch.float32) >= float(min_dist))
    if split == "train":
        mask = mask & data.train_mask
    elif split == "train_val":
        mask = mask & (data.train_mask | data.val_mask)
    elif split == "all_hits":
        pass
    else:
        raise ValueError(f"Unknown residual train split: {split}")

    indices = mask.nonzero(as_tuple=False).view(-1)
    indices = _stratified_subsample(trace, indices, max_pairs)
    return indices


def train_residual_adapter(
    target_embeddings,
    verify_features,
    edge_index,
    trace,
    data,
    risk_scores=None,
    rank=32,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    residual_l2=1e-4,
    train_split="train_val",
    max_pairs=4096,
    correction_mask=None,
    min_dist=0.0,
    controller=None,
    hash_route_features=None,
    extra_anchors_per_node=0,
    extra_query_nodes=0,
    positive_error_max=-1.0,
    adapter_type="low_rank",
    hidden_dim=0,
    hidden_layers=2,
    dropout=0.1,
    accept_mode="shared",
    cosine_weight=1.0,
    mse_weight=0.25,
    delta_weight=0.5,
    bucket_mode="support",
    gate_loss_weight=0.0,
    accept_loss_weight=0.0,
    gate_error_scale=0.25,
    gate_error_max=0.45,
    gate_sparsity_weight=0.0,
    extra_negative_anchors_per_node=0,
    negative_error_min=0.45,
    negative_gate_weight=1.0,
    class_aware_accept=False,
    classifier_accept_gate=False,
    classifier_model=None,
    classifier_reference_logits=None,
    classifier_accept_mode="both",
    classifier_accept_scope="node",
    classifier_accept_after_residual=False,
    classifier_accept_probe_alpha=0.25,
    classifier_accept_max_kl=0.2,
):
    device = target_embeddings.device
    train_nodes = select_residual_train_nodes(
        trace,
        data,
        split=train_split,
        max_pairs=max_pairs,
        correction_mask=correction_mask,
        min_dist=min_dist,
    )
    if train_nodes.numel() == 0:
        return None, {"train_pairs": 0, "loss": 0.0}

    pair_tensors = _build_training_pair_tensors(trace, train_nodes, device)
    pair_tensors, base_pairs_kept, base_pairs_total = _filter_positive_pairs_by_error(
        pair_tensors,
        target_embeddings,
        positive_error_max,
    )
    extra_pairs, extra_pair_stats = _harvest_extra_anchor_pairs(
        controller,
        hash_route_features,
        verify_features,
        target_embeddings,
        trace,
        data,
        train_nodes,
        train_split,
        min_dist,
        extra_anchors_per_node,
        extra_query_nodes,
        max(0, int(max_pairs) - int(train_nodes.numel())),
        positive_error_max,
        bucket_mode=bucket_mode,
    )
    pair_tensors = _append_training_pairs(pair_tensors, extra_pairs)
    accept_targets = None
    class_accept_labelled = 0
    class_accept_positive = 0
    classifier_accept_evaluated = 0
    classifier_accept_positive = 0
    classifier_accept_mean_kl = 0.0
    class_targets = None
    if bool(classifier_accept_gate) and not bool(classifier_accept_after_residual):
        (
            accept_targets,
            classifier_accept_evaluated,
            classifier_accept_positive,
            classifier_accept_mean_kl,
        ) = _build_classifier_accept_targets(
            classifier_model,
            data,
            pair_tensors,
            target_embeddings,
            classifier_reference_logits,
            mode=classifier_accept_mode,
            scope=classifier_accept_scope,
            max_kl=classifier_accept_max_kl,
        )
    if bool(class_aware_accept):
        class_targets, class_accept_labelled, class_accept_positive = _build_class_accept_targets(
            data,
            pair_tensors,
            train_split,
        )
        if class_targets is not None:
            accept_targets = class_targets if accept_targets is None else accept_targets * class_targets
    negative_pairs, negative_pair_stats = _harvest_negative_anchor_pairs(
        controller,
        hash_route_features,
        verify_features,
        target_embeddings,
        trace,
        data,
        train_nodes,
        train_split,
        extra_negative_anchors_per_node,
        extra_query_nodes,
        max(0, int(max_pairs)),
        negative_error_min,
        bucket_mode=bucket_mode,
    )
    accept_targets = None
    class_accept_labelled = 0
    class_accept_positive = 0
    classifier_accept_evaluated = 0
    classifier_accept_positive = 0
    classifier_accept_mean_kl = 0.0
    class_targets = None
    if pair_tensors["node_indices"].numel() == 0:
        return None, {
            "train_pairs": 0,
            "base_train_nodes": int(train_nodes.numel()),
            "base_pairs_kept": int(base_pairs_kept),
            "base_pairs_total": int(base_pairs_total),
            "extra_pairs": 0 if extra_pairs is None else int(extra_pairs["node_indices"].numel()),
            "extra_pair_stats": dict(extra_pair_stats or {}),
            "negative_pairs": 0 if negative_pairs is None else int(negative_pairs["node_indices"].numel()),
            "negative_pair_stats": dict(negative_pair_stats or {}),
            "loss": 0.0,
            "gate_mean": 0.0,
            "accept_gate_mean": 0.0,
            "support_pairs": {},
            "support_losses": {},
            "support_gate_means": {},
            "support_accept_gate_means": {},
            "support_aware": False,
            "adapter_type": str(adapter_type),
            "accept_mode": str(accept_mode),
            "accept_calibration": None,
            "support_accept_calibration": {},
            "bucket_mode": str(bucket_mode),
            "class_aware_accept": bool(class_aware_accept),
            "class_accept_labelled": 0,
            "class_accept_positive": 0,
            "classifier_accept_gate": bool(classifier_accept_gate),
            "classifier_accept_evaluated": 0,
            "classifier_accept_positive": 0,
            "classifier_accept_mean_kl": 0.0,
            "classifier_accept_after_residual": bool(classifier_accept_after_residual),
            "classifier_accept_probe_alpha": float(classifier_accept_probe_alpha),
        }

    if bool(classifier_accept_gate) and not bool(classifier_accept_after_residual):
        (
            accept_targets,
            classifier_accept_evaluated,
            classifier_accept_positive,
            classifier_accept_mean_kl,
        ) = _build_classifier_accept_targets(
            classifier_model,
            data,
            pair_tensors,
            target_embeddings,
            classifier_reference_logits,
            mode=classifier_accept_mode,
            scope=classifier_accept_scope,
            max_kl=classifier_accept_max_kl,
        )
    if bool(class_aware_accept):
        class_targets, class_accept_labelled, class_accept_positive = _build_class_accept_targets(
            data,
            pair_tensors,
            train_split,
        )
        if class_targets is not None:
            accept_targets = class_targets if accept_targets is None else accept_targets * class_targets

    x_train = build_residual_pair_inputs(
        verify_features,
        edge_index,
        trace,
        pair_tensors["node_indices"],
        risk_scores=risk_scores,
        source_ids=pair_tensors["source_ids"],
        best_dists=pair_tensors["best_dists"],
        route_hit_counts=pair_tensors["route_hit_counts"],
        base_route_hit_counts=pair_tensors["base_route_hit_counts"],
        winning_base_table_hit_counts=pair_tensors["winning_base_table_hit_counts"],
        best_cosines=pair_tensors["best_cosines"],
    )
    source_ids = pair_tensors["source_ids"]
    anchors = target_embeddings[source_ids]
    targets = target_embeddings[pair_tensors["node_indices"]]
    x_train_neg = None
    if negative_pairs is not None:
        x_train_neg = build_residual_pair_inputs(
            verify_features,
            edge_index,
            trace,
            negative_pairs["node_indices"],
            risk_scores=risk_scores,
            source_ids=negative_pairs["source_ids"],
            best_dists=negative_pairs["best_dists"],
            route_hit_counts=negative_pairs["route_hit_counts"],
            base_route_hit_counts=negative_pairs["base_route_hit_counts"],
            winning_base_table_hit_counts=negative_pairs["winning_base_table_hit_counts"],
            best_cosines=negative_pairs["best_cosines"],
        )
    if bool(classifier_accept_gate) and bool(classifier_accept_after_residual):
        probe_epochs = max(20, int(epochs) // 2)
        probe_adapter, _probe_loss, _probe_gate_mean, _probe_accept_gate_mean = _fit_residual_adapter(
            x_train,
            anchors,
            targets,
            rank=rank,
            epochs=probe_epochs,
            lr=lr,
            weight_decay=weight_decay,
            residual_l2=residual_l2,
            adapter_type=adapter_type,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            accept_mode=accept_mode,
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            delta_weight=delta_weight,
            gate_loss_weight=gate_loss_weight,
            accept_loss_weight=accept_loss_weight,
            gate_error_scale=gate_error_scale,
            gate_error_max=gate_error_max,
            gate_sparsity_weight=gate_sparsity_weight,
            x_neg=x_train_neg,
            negative_gate_weight=negative_gate_weight,
            accept_targets=class_targets,
        )
        probe_adapter.eval()
        with torch.no_grad():
            probe_residual, _probe_raw_delta, _probe_gate, _probe_accept_gate = _adapter_forward(probe_adapter, x_train)
            probe_candidates = anchors + float(classifier_accept_probe_alpha) * probe_residual
        (
            accept_targets,
            classifier_accept_evaluated,
            classifier_accept_positive,
            classifier_accept_mean_kl,
        ) = _build_classifier_accept_targets(
            classifier_model,
            data,
            pair_tensors,
            target_embeddings,
            classifier_reference_logits,
            mode=classifier_accept_mode,
            scope=classifier_accept_scope,
            max_kl=classifier_accept_max_kl,
            candidate_embeddings=probe_candidates,
        )
        if class_targets is not None:
            accept_targets = accept_targets * class_targets
    global_adapter, global_loss, global_gate_mean, global_accept_gate_mean = _fit_residual_adapter(
        x_train,
        anchors,
        targets,
        rank=rank,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        residual_l2=residual_l2,
        adapter_type=adapter_type,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        accept_mode=accept_mode,
        cosine_weight=cosine_weight,
        mse_weight=mse_weight,
        delta_weight=delta_weight,
        gate_loss_weight=gate_loss_weight,
        accept_loss_weight=accept_loss_weight,
        gate_error_scale=gate_error_scale,
        gate_error_max=gate_error_max,
        gate_sparsity_weight=gate_sparsity_weight,
        x_neg=x_train_neg,
        negative_gate_weight=negative_gate_weight,
        accept_targets=accept_targets,
    )

    support_hits = pair_tensors["winning_base_table_hit_counts"].to(device=device, dtype=torch.long)
    best_dists = pair_tensors["best_dists"].to(device=device, dtype=torch.long)
    bucket_values = compute_bucket_values_from_tensors(support_hits, best_dists, bucket_mode=bucket_mode).to(
        device=device, dtype=torch.long
    )
    if negative_pairs is not None:
        neg_bucket_values = compute_bucket_values_from_tensors(
            negative_pairs["winning_base_table_hit_counts"].to(device=device, dtype=torch.long),
            negative_pairs["best_dists"].to(device=device, dtype=torch.long),
            bucket_mode=bucket_mode,
        ).to(device=device, dtype=torch.long)
    else:
        neg_bucket_values = None
    adapters_by_support = {}
    support_pair_counts = {}
    support_losses = {}
    support_gate_means = {}
    support_accept_gate_means = {}
    support_accept_calibration = {}
    min_bucket_pairs = max(32, int(rank))

    for support_value in sorted(set(int(v) for v in bucket_values.detach().cpu().tolist())):
        bucket_mask = bucket_values == int(support_value)
        bucket_pairs = int(bucket_mask.sum().item())
        if bucket_pairs < min_bucket_pairs:
            continue
        bucket_x = x_train[bucket_mask]
        bucket_anchors = anchors[bucket_mask]
        bucket_targets = targets[bucket_mask]
        bucket_adapter, bucket_loss, bucket_gate_mean, bucket_accept_gate_mean = _fit_residual_adapter(
            bucket_x,
            bucket_anchors,
            bucket_targets,
            rank=rank,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            residual_l2=residual_l2,
            adapter_type=adapter_type,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            accept_mode=accept_mode,
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            delta_weight=delta_weight,
            gate_loss_weight=gate_loss_weight,
            accept_loss_weight=accept_loss_weight,
            gate_error_scale=gate_error_scale,
            gate_error_max=gate_error_max,
            gate_sparsity_weight=gate_sparsity_weight,
            x_neg=None if x_train_neg is None else x_train_neg[neg_bucket_values == int(support_value)],
            negative_gate_weight=negative_gate_weight,
            accept_targets=None if accept_targets is None else accept_targets[bucket_mask],
        )
        adapters_by_support[int(support_value)] = bucket_adapter
        support_pair_counts[int(support_value)] = bucket_pairs
        support_losses[int(support_value)] = bucket_loss
        support_gate_means[int(support_value)] = bucket_gate_mean
        support_accept_gate_means[int(support_value)] = bucket_accept_gate_mean
        calibration_info = _extract_accept_calibration(bucket_adapter)
        if calibration_info is not None:
            support_accept_calibration[int(support_value)] = calibration_info

    if adapters_by_support:
        adapter = SupportAwareResidualAdapter(
            global_adapter,
            adapters_by_support=adapters_by_support,
            bucket_mode=bucket_mode,
        )
    else:
        adapter = global_adapter

    return adapter, {
        "train_pairs": int(pair_tensors["node_indices"].numel()),
        "base_train_nodes": int(train_nodes.numel()),
        "base_pairs_kept": int(base_pairs_kept),
        "base_pairs_total": int(base_pairs_total),
        "extra_pairs": 0 if extra_pairs is None else int(extra_pairs["node_indices"].numel()),
        "extra_pair_stats": dict(extra_pair_stats or {}),
        "negative_pairs": 0 if negative_pairs is None else int(negative_pairs["node_indices"].numel()),
        "negative_pair_stats": dict(negative_pair_stats or {}),
        "loss": global_loss,
        "gate_mean": global_gate_mean,
        "accept_gate_mean": global_accept_gate_mean,
        "support_pairs": support_pair_counts,
        "support_losses": support_losses,
        "support_gate_means": support_gate_means,
        "support_accept_gate_means": support_accept_gate_means,
        "support_aware": bool(adapters_by_support),
        "adapter_type": str(adapter_type),
        "accept_mode": str(accept_mode),
        "accept_calibration": _extract_accept_calibration(global_adapter),
        "support_accept_calibration": support_accept_calibration,
        "bucket_mode": str(bucket_mode),
        "class_aware_accept": bool(class_aware_accept),
        "class_accept_labelled": int(class_accept_labelled),
        "class_accept_positive": int(class_accept_positive),
        "classifier_accept_gate": bool(classifier_accept_gate),
        "classifier_accept_evaluated": int(classifier_accept_evaluated),
        "classifier_accept_positive": int(classifier_accept_positive),
        "classifier_accept_mean_kl": float(classifier_accept_mean_kl),
        "classifier_accept_after_residual": bool(classifier_accept_after_residual),
        "classifier_accept_probe_alpha": float(classifier_accept_probe_alpha),
    }


def apply_residual_adapter(
    direct_embeddings,
    target_embeddings,
    verify_features,
    edge_index,
    trace,
    adapter,
    risk_scores=None,
    alpha=1.0,
    gate_accept_threshold=None,
    min_dist=1.0,
    correction_mask=None,
    normalize_corrected=False,
):
    if adapter is None:
        return direct_embeddings, {
            "corrected": 0,
            "alpha": 0.0,
            "gate": 0.0,
            "accept_gate": 1.0,
            "accepted": 0,
            "rejected": 0,
        }
    device = direct_embeddings.device
    hit_nodes = (trace["hit_mask"] & (trace["source_ids"] >= 0)).nonzero(as_tuple=False).view(-1)
    if hit_nodes.numel() == 0:
        return direct_embeddings, {"corrected": 0, "alpha": 0.0, "gate": 0.0, "accepted": 0, "rejected": 0}

    if float(min_dist) > 0.0:
        dist = trace["best_dists"][hit_nodes].to(device=device, dtype=torch.float32)
        hit_nodes = hit_nodes[dist >= float(min_dist)]
        if hit_nodes.numel() == 0:
            return direct_embeddings, {
                "corrected": 0,
                "alpha": float(alpha),
                "gate": 0.0,
                "accept_gate": 1.0,
                "accepted": 0,
                "rejected": 0,
            }
    if correction_mask is not None:
        active = correction_mask.to(device=device, dtype=torch.bool)
        hit_nodes = hit_nodes[active[hit_nodes]]
        if hit_nodes.numel() == 0:
            return direct_embeddings, {
                "corrected": 0,
                "alpha": float(alpha),
                "gate": 0.0,
                "accept_gate": 1.0,
                "accepted": 0,
                "rejected": 0,
            }

    corrected = direct_embeddings.clone()
    support_hits = trace["winning_base_table_hit_counts"][hit_nodes].to(device=device, dtype=torch.long)
    best_dists = trace["best_dists"][hit_nodes].to(device=device, dtype=torch.long)
    alpha_map = None
    default_alpha = None
    if isinstance(alpha, dict):
        alpha_map = _as_float_dict(alpha.get("by_support", alpha))
        default_alpha = float(alpha.get("default", 0.0))
    else:
        default_alpha = float(alpha)

    gate_threshold = None if gate_accept_threshold is None else float(gate_accept_threshold)

    def _apply_with(local_adapter, node_subset, local_alpha):
        local_adapter.eval()
        with torch.no_grad():
            x = build_residual_pair_inputs(verify_features, edge_index, trace, node_subset, risk_scores=risk_scores)
            source_ids = trace["source_ids"][node_subset].to(device=device, dtype=torch.long)
            anchors = target_embeddings[source_ids]
            residual, _raw_delta, correction_gate, accept_gate = _adapter_forward(local_adapter, x)
            if accept_gate is None or gate_threshold is None or gate_threshold <= 0.0:
                accept_mask = torch.ones(node_subset.numel(), dtype=torch.bool, device=device)
            else:
                accept_mask = accept_gate.view(-1) >= gate_threshold
            reject_mask = ~accept_mask
            if bool(accept_mask.any().item()):
                accept_nodes = node_subset[accept_mask]
                corrected_values = anchors[accept_mask] + float(local_alpha) * residual[accept_mask]
                if bool(normalize_corrected):
                    corrected_values = F.normalize(corrected_values, p=2, dim=1)
                corrected[accept_nodes] = corrected_values
            if bool(reject_mask.any().item()):
                reject_nodes = node_subset[reject_mask]
                corrected[reject_nodes] = target_embeddings[reject_nodes]
        correction_gate_mean = 1.0 if correction_gate is None else float(correction_gate.mean().item())
        accept_gate_mean = 1.0 if accept_gate is None else float(accept_gate.mean().item())
        return (
            correction_gate_mean,
            accept_gate_mean,
            int(accept_mask.sum().item()),
            int(reject_mask.sum().item()),
            node_subset[reject_mask],
        )

    support_alpha_used = {}
    support_gate_used = {}
    support_accept_gate_used = {}
    weighted_alpha = 0.0
    weighted_count = 0
    weighted_gate = 0.0
    weighted_accept_gate = 0.0
    total_accepted = 0
    total_rejected = 0
    rejected_nodes = []
    if isinstance(adapter, SupportAwareResidualAdapter):
        bucket_mode = adapter.bucket_mode
    else:
        bucket_mode = "support"
    bucket_values = compute_bucket_values_from_tensors(support_hits, best_dists, bucket_mode=bucket_mode).to(
        device=device, dtype=torch.long
    )
    unique_supports = sorted(set(int(v) for v in bucket_values.detach().cpu().tolist()))
    use_support_loop = isinstance(adapter, SupportAwareResidualAdapter) or alpha_map is not None

    if use_support_loop:
        for support_value in unique_supports:
            node_subset = hit_nodes[bucket_values == int(support_value)]
            if isinstance(adapter, SupportAwareResidualAdapter):
                local_adapter = adapter.adapters_by_support.get(int(support_value), adapter.global_adapter)
                support_adapters = len(adapter.adapters_by_support)
            else:
                local_adapter = adapter
                support_adapters = 0
            local_alpha = default_alpha if alpha_map is None else float(alpha_map.get(int(support_value), default_alpha))
            support_alpha_used[int(support_value)] = float(local_alpha)
            correction_gate_mean, accept_gate_mean, accepted_count, rejected_count, rejected_subset = _apply_with(
                local_adapter, node_subset, local_alpha
            )
            support_gate_used[int(support_value)] = float(correction_gate_mean)
            support_accept_gate_used[int(support_value)] = float(accept_gate_mean)
            weighted_alpha += float(local_alpha) * int(node_subset.numel())
            weighted_count += int(node_subset.numel())
            weighted_gate += float(correction_gate_mean) * int(node_subset.numel())
            weighted_accept_gate += float(accept_gate_mean) * int(node_subset.numel())
            total_accepted += int(accepted_count)
            total_rejected += int(rejected_count)
            if int(rejected_subset.numel()) > 0:
                rejected_nodes.append(rejected_subset)
        return corrected, {
            "corrected": int(hit_nodes.numel()),
            "alpha": 0.0 if weighted_count == 0 else float(weighted_alpha / max(1, weighted_count)),
            "gate": 0.0 if weighted_count == 0 else float(weighted_gate / max(1, weighted_count)),
            "accept_gate": 1.0 if weighted_count == 0 else float(weighted_accept_gate / max(1, weighted_count)),
            "alpha_by_support": support_alpha_used,
            "gate_by_support": support_gate_used,
            "accept_gate_by_support": support_accept_gate_used,
            "accepted": int(total_accepted),
            "rejected": int(total_rejected),
            "rejected_nodes": torch.cat(rejected_nodes, dim=0) if rejected_nodes else torch.empty(0, dtype=torch.long, device=device),
            "gate_accept_threshold": gate_threshold,
            "support_adapters": support_adapters,
        }

    correction_gate_mean, accept_gate_mean, accepted_count, rejected_count, rejected_subset = _apply_with(
        adapter, hit_nodes, default_alpha
    )
    return corrected, {
        "corrected": int(hit_nodes.numel()),
        "alpha": float(default_alpha),
        "gate": float(correction_gate_mean),
        "accept_gate": float(accept_gate_mean),
        "accepted": int(accepted_count),
        "rejected": int(rejected_count),
        "rejected_nodes": rejected_subset,
        "gate_accept_threshold": gate_threshold,
    }


def embedding_error(reference_embeddings, approx_embeddings):
    return (1.0 - F.cosine_similarity(reference_embeddings, approx_embeddings, dim=1)).clamp(min=0.0)
