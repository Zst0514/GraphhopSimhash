import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadHashProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_heads)
        )

    def reset_parameters(self, base_seed):
        for head_idx, head in enumerate(self.heads):
            local_seed = int(base_seed) + head_idx * 9176
            torch.manual_seed(local_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(local_seed)
            nn.init.orthogonal_(head.weight)

    def project_head(self, x, head_idx):
        return F.normalize(self.heads[head_idx](x), p=2, dim=-1)

    def project_all(self, x):
        return [self.project_head(x, head_idx) for head_idx in range(len(self.heads))]


def build_hash_random_matrix(input_dim, sketch_bits, device, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(input_dim, sketch_bits, generator=generator)
    random_matrix = F.normalize(random_matrix, p=2, dim=0)
    return random_matrix.to(device)
def collect_learned_hash_pairs(
    features,
    oracle_embs,
    oracle_logits,
    supervision_mask,
    max_nodes=2048,
    topk=48,
    pos_per_anchor=4,
    neg_per_anchor=8,
    pos_tau=0.95,
    neg_tau=0.85,
):
    supervision_idx = supervision_mask.nonzero(as_tuple=False).view(-1)
    if supervision_idx.numel() == 0:
        return None
    if max_nodes is not None and supervision_idx.numel() > max_nodes:
        supervision_idx = supervision_idx[:max_nodes]

    support_features = features[supervision_idx]
    support_oracle = F.normalize(oracle_embs[supervision_idx], p=2, dim=1)
    input_cos = support_features @ support_features.T
    oracle_cos = support_oracle @ support_oracle.T

    topk = min(int(topk), supervision_idx.numel())
    if topk <= 1:
        return None
    neighbor_ids = torch.topk(input_cos, k=topk, dim=1).indices[:, 1:]

    class_ids = None
    if oracle_logits is not None:
        class_ids = oracle_logits[supervision_idx].argmax(dim=1)
        class_cpu = class_ids.cpu()
    else:
        class_cpu = None

    oracle_cos_cpu = oracle_cos.cpu()
    pair_i = []
    pair_j = []
    labels = []

    for local_i in range(support_features.size(0)):
        candidate_ids = neighbor_ids[local_i].tolist()
        pos_added = 0
        neg_added = 0

        def same_class(local_j):
            if class_cpu is None:
                return True
            return int(class_cpu[local_i].item()) == int(class_cpu[local_j].item())

        for local_j in candidate_ids:
            emb_sim = float(oracle_cos_cpu[local_i, local_j].item())
            if same_class(local_j) and emb_sim >= pos_tau and pos_added < pos_per_anchor:
                pair_i.append(local_i)
                pair_j.append(local_j)
                labels.append(1.0)
                pos_added += 1
            elif ((not same_class(local_j)) or emb_sim <= neg_tau) and neg_added < neg_per_anchor:
                pair_i.append(local_i)
                pair_j.append(local_j)
                labels.append(0.0)
                neg_added += 1
            if pos_added >= pos_per_anchor and neg_added >= neg_per_anchor:
                break

        if pos_added == 0:
            relaxed_pos_tau = max(0.0, pos_tau - 0.05)
            for local_j in candidate_ids:
                emb_sim = float(oracle_cos_cpu[local_i, local_j].item())
                if same_class(local_j) and emb_sim >= relaxed_pos_tau:
                    pair_i.append(local_i)
                    pair_j.append(local_j)
                    labels.append(1.0)
                    break

        if neg_added == 0:
            relaxed_neg_tau = min(1.0, neg_tau + 0.05)
            for local_j in candidate_ids:
                emb_sim = float(oracle_cos_cpu[local_i, local_j].item())
                if (not same_class(local_j)) or emb_sim <= relaxed_neg_tau:
                    pair_i.append(local_i)
                    pair_j.append(local_j)
                    labels.append(0.0)
                    break

    if not labels:
        return None

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    if labels_tensor.sum().item() <= 0 or labels_tensor.sum().item() >= labels_tensor.numel():
        return None

    return {
        "support_idx": supervision_idx,
        "pair_i": torch.tensor(pair_i, dtype=torch.long),
        "pair_j": torch.tensor(pair_j, dtype=torch.long),
        "labels": labels_tensor,
    }


def _resolve_route_head_bit_schedule(args, route_spec):
    default_num_heads = max(1, int(getattr(args, "hash_heads_per_route", 1)))
    global_schedule = getattr(args, "hash_head_bits", None)
    main_schedule = getattr(args, "main_hash_head_bits", None)
    union_schedule = getattr(args, "union_hash_head_bits", None)
    topology_schedule = getattr(args, "topology_hash_head_bits", None)
    route_role = str(route_spec.get("route_role", "semantic"))
    base_route_idx = int(route_spec.get("base_route_idx", 0))

    if route_role == "topology" and topology_schedule is not None:
        schedule = [int(bits) for bits in topology_schedule]
    elif base_route_idx == 0 and main_schedule is not None:
        schedule = [int(bits) for bits in main_schedule]
    elif base_route_idx > 0 and union_schedule is not None:
        schedule = [int(bits) for bits in union_schedule]
    elif global_schedule is not None:
        schedule = [int(bits) for bits in global_schedule]
    else:
        schedule = [int(args.sketch_bits)] * default_num_heads

    if len(schedule) == 1 and default_num_heads > 1:
        schedule = schedule * default_num_heads
    return schedule


def _expand_raw_multihead_hash_routes(route_specs, args, device):
    head_seed = int(getattr(args, "hash_head_seed", 12345))
    expanded_route_specs = []

    for spec in route_specs:
        head_bit_schedule = _resolve_route_head_bit_schedule(args, spec)
        num_heads = len(head_bit_schedule)
        input_dim = int(spec["features"].size(1))
        encoder_seed = head_seed + int(spec.get("base_route_idx", 0)) * 1009

        for head_idx, head_bits in enumerate(head_bit_schedule):
            hash_seed = int(encoder_seed + head_idx * 9176)
            expanded_route_specs.append(
                {
                    **spec,
                    "name": f"{spec['name']}__raw__b{int(head_bits)}__head{head_idx}",
                    "features": spec["features"].detach(),
                    "hash_matrix": build_hash_random_matrix(
                        input_dim,
                        int(head_bits),
                        device,
                        hash_seed,
                    ),
                    "hash_bits": int(head_bits),
                    "table_idx": head_idx,
                    "table_count": num_heads,
                }
            )

    return expanded_route_specs, None


def fit_multihead_hash_projection(route_specs, oracle_embs, oracle_logits, supervision_mask, args, device):
    if not args.learned_hash_projection:
        return _expand_raw_multihead_hash_routes(route_specs, args, device)

    head_seed = int(getattr(args, "hash_head_seed", 12345))
    projected_route_specs = []
    route_stats = []
    trained_encoders = 0
    route_head_schedules = {}
    total_heads = 0
    shared_projection_heads = bool(getattr(args, "shared_hash_projection_heads", False))

    for spec in route_specs:
        base_route_idx = int(spec.get("base_route_idx", 0))
        head_bit_schedule = _resolve_route_head_bit_schedule(args, spec)
        num_heads = len(head_bit_schedule)
        route_head_schedules[spec["name"]] = list(head_bit_schedule)
        total_heads += num_heads

        pair_data = collect_learned_hash_pairs(
            spec["features"],
            oracle_embs,
            oracle_logits,
            supervision_mask=supervision_mask,
            max_nodes=args.learned_hash_supervision_limit,
            topk=args.learned_hash_topk,
            pos_per_anchor=args.learned_hash_pos_per_anchor,
            neg_per_anchor=args.learned_hash_neg_per_anchor,
            pos_tau=args.learned_hash_pos_tau,
            neg_tau=args.learned_hash_neg_tau,
        )

        input_dim = spec["features"].size(1)
        encoder_seed = head_seed + int(spec.get("base_route_idx", 0)) * 1009
        projection_head_count = 1 if shared_projection_heads else num_heads
        encoder = MultiHeadHashProjection(input_dim, args.learned_hash_dim, projection_head_count).to(device)
        encoder.reset_parameters(encoder_seed)

        per_projection_head_losses = [None] * projection_head_count
        if pair_data is not None:
            support_idx = pair_data["support_idx"]
            pair_i = pair_data["pair_i"].to(device)
            pair_j = pair_data["pair_j"].to(device)
            labels = pair_data["labels"].to(device)
            train_features = spec["features"][support_idx].to(device)
            batch_size = min(args.learned_hash_batch_size, labels.numel())

            # The default keeps independent learned heads for diversity. The
            # shared mode is lighter: one projection per route, multiple hash
            # heads only differ by random hash seed and bit width.
            for projection_head_idx in range(projection_head_count):
                optimizer = torch.optim.AdamW(
                    encoder.heads[projection_head_idx].parameters(),
                    lr=args.learned_hash_lr,
                    weight_decay=args.learned_hash_weight_decay,
                )

                epoch_loss = None
                for _ in range(args.learned_hash_epochs):
                    perm = torch.randperm(labels.numel(), device=device)
                    head_loss_sum = 0.0
                    seen = 0
                    for start in range(0, labels.numel(), batch_size):
                        batch_ids = perm[start:start + batch_size]
                        batch_i = pair_i[batch_ids]
                        batch_j = pair_j[batch_ids]
                        batch_labels = labels[batch_ids]

                        proj_i = encoder.project_head(train_features[batch_i], projection_head_idx)
                        proj_j = encoder.project_head(train_features[batch_j], projection_head_idx)
                        pair_cos = (proj_i * proj_j).sum(dim=1)

                        pos_mask = batch_labels > 0.5
                        neg_mask = ~pos_mask

                        loss = torch.tensor(0.0, device=device)
                        if pos_mask.any():
                            loss = loss + (1.0 - pair_cos[pos_mask]).mean()
                        if neg_mask.any():
                            loss = loss + F.relu(pair_cos[neg_mask] - args.learned_hash_neg_margin).mean()

                        unique_ids = torch.unique(torch.cat([batch_i, batch_j], dim=0))
                        proj_unique = encoder.project_head(train_features[unique_ids], projection_head_idx)
                        balance_loss = proj_unique.mean(dim=0).pow(2).mean()
                        loss = loss + args.learned_hash_balance_lambda * balance_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batch_count = batch_ids.numel()
                        head_loss_sum += float(loss.item()) * batch_count
                        seen += batch_count

                    epoch_loss = head_loss_sum / max(1, seen)

                per_projection_head_losses[projection_head_idx] = epoch_loss
            trained_encoders += 1
            support_nodes = int(support_idx.numel())
            pair_count = int(labels.numel())
            positive_rate = float(labels.mean().item())
            trained = True
        else:
            support_nodes = 0
            pair_count = 0
            positive_rate = 0.0
            trained = False

        encoder.eval()
        with torch.no_grad():
            projected_heads = encoder.project_all(spec["features"])
        if shared_projection_heads:
            projected_heads = projected_heads * num_heads
            per_head_losses = per_projection_head_losses * num_heads
        else:
            per_head_losses = per_projection_head_losses

        for head_idx, head_features in enumerate(projected_heads):
            hash_seed = int(encoder_seed + head_idx * 9176)
            head_bits = int(head_bit_schedule[head_idx])
            projected_route_specs.append(
                {
                    **spec,
                    "name": f"{spec['name']}__mh{args.learned_hash_dim}__b{head_bits}__head{head_idx}",
                    "features": head_features.detach(),
                    "hash_matrix": build_hash_random_matrix(
                        args.learned_hash_dim,
                        head_bits,
                        device,
                        hash_seed,
                    ),
                    "hash_bits": head_bits,
                    "table_idx": head_idx,
                    "table_count": num_heads,
                }
            )
            route_stats.append(
                {
                    "route_name": spec["name"],
                    "head_idx": head_idx,
                    "trained": trained,
                    "support_nodes": support_nodes,
                    "pair_count": pair_count,
                    "positive_rate": positive_rate,
                    "train_loss": per_head_losses[head_idx] if per_head_losses[head_idx] is not None else None,
                    "shared_projection": shared_projection_heads,
                    "projection_head_idx": 0 if shared_projection_heads else head_idx,
                    "seed": int(encoder_seed + head_idx * 9176),
                    "hash_seed": hash_seed,
                    "hash_bits": head_bits,
                }
            )

    stats = {
        "trained_encoders": trained_encoders,
        "total_encoders": len(route_specs),
        "total_heads": total_heads,
        "route_head_schedules": route_head_schedules,
        "routes": route_stats,
        "output_dim": int(args.learned_hash_dim),
        "shared_projection_heads": shared_projection_heads,
    }
    return projected_route_specs, stats
