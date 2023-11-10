import os
import time
import torch
from pathlib import Path
import thop
import copy
import json
import random
from tqdm import trange, tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

from performance_predictor.predictor import JsonData, RFPredictor


def sample_subnet(supernet, specs=None, arch_cfg=None, existing_archs=None, flops_limits=None, device=torch.device("cuda:0")):

    inputs = torch.randn(1, 3, 32, 32).to(device)  # for thop to compute flops

    while True:
        if specs == "min":
            subnet_cfg = supernet.sample_min_subnet()
        elif specs == "max":
            subnet_cfg = supernet.sample_max_subnet()
        elif specs == "mutate":
            subnet_cfg = supernet.mutate_and_reset(arch_cfg, prob=0.2, keep_resolution=True)  # TODO(Variant): value?
        elif specs == "crossover":
            parents = random.choices(arch_cfg, k=2)
            arch_cfg_A, arch_cfg_B = copy.deepcopy(parents[0]), copy.deepcopy(parents[1])
            subnet_cfg = supernet.crossover_and_reset(arch_cfg_A, arch_cfg_B, p=0.5)  # TODO(Variant): value?
        elif specs == "random":
            subnet_cfg = supernet.sample_active_subnet(compute_flops=False)
        else:
            raise ValueError

        subnet = supernet.get_active_subnet(preserve_weight=False)
        flops, params = thop.profile(subnet, inputs=(inputs,), verbose=False)

        if existing_archs is not None:
            if subnet_cfg in existing_archs:  # 若已存在该架构
                continue
        if flops_limits is not None:
            if flops > flops_limits:  # 若超出硬件约束
                continue

        break

    return {"arch_cfg": subnet_cfg, "flops": flops, "params": params, "minmax": specs in ['min', "max"]}


def create_and_evaluate(cfg, supernet, data, subnet_info, finetune=False, inplace=True):
    supernet.set_active_subnet(**subnet_info["arch_cfg"])

    subnet = supernet.get_active_subnet(preserve_weight=True)
    evaluate_results = get_fitness(cfg, subnet, data, finetune=finetune)

    subnet_info.update(evaluate_results)  # inplace update

    return subnet_info


def evolution_search(cfg, supernet, data, flops_limit, max_generations=20, popu_size=256, cid=0):

    population = []
    saved_infos = []

    # EA 初始化 population ----------------------------------------------------------------------------------------------

    # 添加min_subnet到初始化的EA population中-----------------------------------------------------------------------------
    subnet_info = sample_subnet(supernet, specs="min", flops_limits=flops_limit, device=cfg.device)
    # 构建min_subnet评估fitness
    subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info, finetune=False)
    # 保存数据
    subnet_info.update({"init_pops", True})
    saved_infos.append(subnet_info)
    population.append(copy.deepcopy(subnet_info['arch_cfg']))

    # 添加其他random subnet到初始化的EA population中----------------------------------------------------------------------
    for _ in trange(popu_size - 1):
        subnet_info = sample_subnet(supernet, specs="random", flops_limits=flops_limit, device=cfg.device)
        # 构建subnet评估fitness
        subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info, finetune=False)
        # 保存数据
        saved_infos.append(subnet_info)
        population.append(copy.deepcopy(subnet_info["arch_cfg"]))  # update populations

    # 构建性能预测器------------------------------------------------------------------------------------------------------

    with open(f"{cfg.outdir}/current_client_init_popu_infos.json", "w") as f:
        json.dump(saved_infos, f, indent=4)

    raw_data = JsonData(f"{cfg.outdir}/current_client_init_popu_infos.json", imbalance_info=None)
    os.remove(f"{cfg.outdir}/current_client_init_popu_infos.json")

    X, Y, Y_aux, target_names, aux_target_names = raw_data.get_ml_dataset(target_names=["loss"],
                                                                          aux_target_names=['flops', 'params'])
    predictor = RFPredictor(max_depth=15, max_features='auto', criterion="squared_error", random_state=0)
    predictor.fit(X, Y)

    # 利用构建的predictor预测初始化population的性能------------------------------------------------------------------------
    # TODO(Variant): 不确定是否对于原始population使用预测的性能还是真实的性能
    for subnet_info in saved_infos:
        subnet_info.update({"pred_loss": subnet_info['server_recalibrate_bn_loss']})

    for current_generation in range(max_generations):
        print(f"current_generation: {current_generation}")
        # 进化世代 实现mutate -------------------------------------------------------------------------------------------
        for _ in trange(int(popu_size/4)):
            parent_arch_cfg = copy.deepcopy(random.choices(population[:popu_size], k=1))[0]
            subnet_info = sample_subnet(supernet, specs="mutate", arch_cfg=parent_arch_cfg,
                                        existing_archs=population, flops_limits=flops_limit, device=cfg.device)
            # 利用构建的predictor预测性能，并将新生成的架构加入所有信息
            subnet_info.update({"pred_loss": predictor.scratch_predict(subnet_info["arch_cfg"])})
            # 保存数据
            saved_infos.append(subnet_info)
            population.append(copy.deepcopy(subnet_info['arch_cfg']))
        # 进化世代 实现crossover ----------------------------------------------------------------------------------------
        for _ in trange(int(popu_size/4)):
            # parents = random.choices(population[:popu_size], k=2)
            # arch_cfg_A, arch_cfg_B = copy.deepcopy(parents[0]), copy.deepcopy(parents[1])
            subnet_info = sample_subnet(supernet, specs="crossover", arch_cfg=population[:popu_size],
                                        existing_archs=population, flops_limits=flops_limit, device=cfg.device)
            # 利用构建的predictor预测性能，并将新生成的架构加入所有信息
            subnet_info.update({"pred_loss": predictor.scratch_predict(subnet_info["arch_cfg"])})
            # 保存数据
            saved_infos.append(subnet_info)
            population.append(copy.deepcopy(subnet_info['arch_cfg']))

        # 进化生成新的种群（排序即可） -------------------------------------------------------------------------------------
        saved_infos.sort(key=lambda x: x['pred_loss'])
        population = [copy.deepcopy(subnet_info['arch_cfg']) for subnet_info in saved_infos]

    # 对last populations架构作精细化finetune评比，并保存相关模型作为后续retrain阶段的预训练模型 -------------------------------
    for i in trange(popu_size):
        saved_infos[i] = create_and_evaluate(cfg, supernet, data, saved_infos[i], finetune=False)
        saved_infos[i].update({"last_pops": True})
        # # 保存模型
        # Path(cfg.outdir + f'/client_{cid}/checkpoints/').mkdir(parents=True, exist_ok=True)
        # subnet_checkpoint_path = os.path.join(cfg.outdir, f"/client_{cid}/checkpoints/", f"subnet{i}.pth")
        # torch.save({"arch_cfg": saved_infos[i]['arch_cfg'], "model": model.state_dict()},
        #            subnet_checkpoint_path)

    saved_infos.sort(key=lambda x: x.get('server_recalibrate_bn_loss', float('inf')))

    # 保存轨迹上所有arch的信息
    with open(f"{cfg.outdir}/client{cid}_popu_infos.json", "w") as f:
        json.dump(saved_infos, f, indent=4)

    return saved_infos


def random_search(cfg, supernet, data, flops_limit, popu_size=200, max_trials=2000):

    population = []
    saved_infos = []

    # 添加min_subnet-----------------------------------------------------------------------------------------------------
    subnet_info = sample_subnet(supernet, specs="min", flops_limits=flops_limit, device=cfg.device)
    # 构建min_subnet评估fitness
    subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info, finetune=False)
    # 保存数据
    saved_infos.append(subnet_info)
    population.append(copy.deepcopy(subnet_info['arch_cfg']))

    # 随机采样若干架构----------------------------------------------------------------------------------------------------
    for _ in trange(max_trials - 1):
        subnet_info = sample_subnet(supernet, specs="random",
                                    existing_archs=population, flops_limits=flops_limit, device=cfg.device)
        # 构建subnet评估fitness
        subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info, finetune=False)
        # 保存数据
        saved_infos.append(subnet_info)
        population.append(copy.deepcopy(subnet_info["arch_cfg"]))  # update populations

    saved_infos.sort(key=lambda x: x['server_recalibrate_bn_loss'])

    # 对top-200架构作精细化finetune评比-----------------------------------------------------------------------------------
    for i in trange(popu_size):
        # 构建subnet评估fitness
        saved_infos[i] = create_and_evaluate(cfg, supernet, data, saved_infos[i], finetune=True)

    saved_infos.sort(key=lambda x: x.get('finetune_fit_loss', float('inf')))

    return saved_infos


def random_explore(cfg, supernet, data, max_trials=2000):

    population = []
    saved_infos = []

    # 添加min_subnet-----------------------------------------------------------------------------
    subnet_info = sample_subnet(supernet, specs="min", device=cfg.device)
    # 构建min_subnet评估fitness
    subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info, finetune=True)
    # 保存数据
    saved_infos.append(subnet_info)
    population.append(copy.deepcopy(subnet_info["arch_cfg"]))

    # 添加max_subnet-----------------------------------------------------------------------------
    subnet_info = sample_subnet(supernet, specs="max", device=cfg.device)
    # 构建max_subnet评估fitness
    subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info["arch_cfg"], finetune=True)
    # 保存数据
    saved_infos.append(subnet_info)
    population.append(copy.deepcopy(subnet_info["arch_cfg"]))

    # 随机采样若干架构----------------------------------------------------------------------------------------------------
    for _ in trange(max_trials - 1):
        subnet_info = sample_subnet(supernet, specs="random", existing_archs=population, device=cfg.device)
        # 构建subnet评估fitness
        subnet_info = create_and_evaluate(cfg, supernet, data, subnet_info, finetune=True)
        # 保存数据
        saved_infos.append(subnet_info)
        population.append(copy.deepcopy(subnet_info["arch_cfg"]))  # update populations

    saved_infos.sort(key=lambda x: x['flops'])

    return saved_infos


def evaluate_model(cfg, model, data_loader):
    n_sample = 0
    n_pos = 0
    loss = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            with autocast(enabled=cfg.use_amp):
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                outputs = model(inputs)
                loss += F.cross_entropy(outputs, targets, label_smoothing=0).item()

            preds = torch.argmax(outputs, dim=-1)
            n_sample += len(preds)
            n_pos += sum(preds == targets).item()

    return loss / n_sample, n_pos / n_sample


def get_fitness(cfg, model, data, finetune=False):
    # ------------------------------------------------------------------------------------------------------------------
    evaluation_results = {}
    # ------------------------------------------------------------------------------------------------------------------

    # NOTE(Variant): recalibrate bn, 校准batchnorm统计，设置model.eval() but batchnorm.training=True.
    # 利用校准后的batchnorm statistics评估模型
    torch.optim.swa_utils.update_bn(data['train'], model, device=cfg.device)
    model.eval()
    client_recalibrate_bn_loss, client_recalibrate_bn_acc = evaluate_model(cfg, model, data["test"])
    evaluation_results.update({'client_recalibrate_bn_loss': client_recalibrate_bn_loss,
                               'client_recalibrate_bn_acc': client_recalibrate_bn_acc})

    # ------------------------------------------------------------------------------------------------------------------

    # NOTE(Variant): recalibrate bn, 校准batchnorm统计，设置model.eval() but batchnorm.training=True.
    # 利用校准后的batchnorm statistics评估模型
    torch.optim.swa_utils.update_bn(data['server'], model, device=cfg.device)
    model.eval()
    server_recalibrate_bn_loss, server_recalibrate_bn_acc = evaluate_model(cfg, model, data["test"])
    evaluation_results.update({'server_recalibrate_bn_loss': server_recalibrate_bn_loss,
                               'server_recalibrate_bn_acc': server_recalibrate_bn_acc})

    # ------------------------------------------------------------------------------------------------------------------
    # 执行finetune
    finetune_epoch = cfg['finetune'].local_update_steps if finetune else 0
    cfg['finetune'].scheduler['max_iters'] = 1

    if finetune_epoch > 0:
        optimizer = get_optimizer(model, **cfg['finetune'].optimizer)  # 创建 finetune optimizer
        scheduler = get_scheduler(optimizer, **cfg['finetune'].scheduler)  # 创建 finetune scheduler

        # prepare mixed precision computation
        scaler = GradScaler() if cfg.use_amp else None

        model.set_bn_param(0.1, 1e-5)
        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
        #         m.reset_running_stats()

        # finetune
        model.train()
        for epoch in range(finetune_epoch):
            for i, (inputs, targets) in enumerate(data['train']):  # finetune dataloader
                with autocast(enabled=cfg.use_amp):
                    inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                    outputs = model(inputs)
                    loss = ctx.criterion(outputs, targets)

                optimizer.zero_grad()

                if cfg.use_amp:
                    scaler.scale(loss).backward()
                    if cfg.grad.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad.grad_clip)
                    optimizer.step()
            scheduler.step()

        # NOTE(Variant): finetune后评估模型
        model.eval()
        finetune_fit_loss, finetune_fit_acc = evaluate_model(cfg, model, data["test"])
        evaluation_results.update({"finetune_fit_loss": finetune_fit_loss, "finetune_fit_acc": finetune_fit_acc})

    return evaluation_results

