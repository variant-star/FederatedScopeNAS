import torch
from torch.cuda.amp import autocast


def calculate_ensemble_logits(inputs, ensemble_models, use_amp, distillation_logits_type):
    """

    Args:
        ensemble_models: List[(weight, model), ......]
        use_amp: torch.cuda.amp.autocast
        logits_type: "max_logits", "min_logits", "majority_vote"

    Returns: logits or fake_labels

    """
    # caculate ensemble distillation logits
    with torch.no_grad():
        y_logits, weights = [], []  # NOTE(Variant): actually, here 'y_true' means 'y_logits'
        for local_weight, client_model in ensemble_models:
            client_model.eval()
            with autocast(enabled=use_amp):
                local_y_logits = client_model(inputs)
            y_logits.append(local_y_logits)
            weights.append(local_weight)

        if distillation_logits_type == "max_logits":
            y_logits = torch.max(torch.stack(y_logits), dim=0)[0]
        elif distillation_logits_type == "avg_logits":
            y_logits = sum([w * v for w, v in zip(weights, y_logits)])

        y_logits = y_logits.clone().detach()
    return y_logits
    # end