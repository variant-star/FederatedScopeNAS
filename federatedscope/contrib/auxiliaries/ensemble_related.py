import torch
from torch.cuda.amp import autocast


@torch.no_grad()
def calculate_ensemble_logits(inputs, ensemble_models, logits_type):
    """

    Args:
        ensemble_models: List[(weight, model), ......]
        logits_type: "max_logits", "min_logits", "majority_vote"

    Returns: logits or fake_labels

    """
    # calculate ensemble distillation logits
    y_logits, weights = [], []  # NOTE(Variant): actually, here 'y_true' means 'y_logits'
    for local_weight, client_model in ensemble_models:
        client_model.eval()
        with autocast(enabled=True):
            local_y_logits = client_model(inputs)
        y_logits.append(local_y_logits)
        weights.append(local_weight)

    if logits_type == "max_logits":
        y_logits = torch.max(torch.stack(y_logits), dim=0)[0]
    elif logits_type == "avg_logits":
        y_logits = sum([w * v for w, v in zip(weights, y_logits)])

    y_logits = y_logits.clone().detach()
    return y_logits