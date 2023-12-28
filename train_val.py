import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from AttentiveNAS.utils.progress import AverageMeter, ProgressMeter, accuracy


def train_one_epoch(model, loader, criterion, optimizer, scaler=None, device=torch.device('cpu')):
    use_amp = scaler is not None
    model.to(device)
    model.train()

    objs = AverageMeter('loss', '6.2f:')
    top1 = AverageMeter('Acc1', '.4f:')
    top5 = AverageMeter('Acc5', '.4f:')

    pbar = tqdm(loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        pbar.set_description(f'TRAIN')
        pbar.set_postfix(loss=objs.avg, top1=top1.avg, top5=top5.avg)
    return objs.avg, top1.avg, top5.avg


@torch.no_grad()
def evaluate_one_epoch(model, loader, use_amp=False, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    objs = AverageMeter('loss', '6.2f:')
    top1 = AverageMeter('Acc1', '.4f:')
    top5 = AverageMeter('Acc5', '.4f:')

    pbar = tqdm(loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        pbar.set_description(f"TEST")
        pbar.set_postfix(loss=objs.avg, top1=top1.avg, top5=top5.avg)

    return objs.avg, top1.avg, top5.avg