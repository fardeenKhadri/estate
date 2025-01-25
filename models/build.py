import os
import models
import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from utils.time_watch import TimeWatch
from models.other.optimizer import build_optimizer
from models.other.criterion import build_criterion


def build_model(config, logger):
    name = config.MODEL.NAME
    w = TimeWatch(f"Build model: {name}", logger)

    ddp = config.WORLD_SIZE > 1
    if ddp:
        logger.info(f"Using Distributed Data Parallel (DDP)")
        dist.init_process_group("nccl", init_method='tcp://127.0.0.1:23456', rank=config.LOCAL_RANK,
                                world_size=config.WORLD_SIZE)

    device = config.TRAIN.DEVICE
    logger.info(f"Creating model: {name} on device: {device}, args: {config.MODEL.ARGS[0]}")

    net = getattr(models, name)
    ckpt_dir = os.path.abspath(os.path.join(config.CKPT.DIR, os.pardir)) if config.DEBUG else config.CKPT.DIR
    model = net(ckpt_dir=ckpt_dir, **config.MODEL.ARGS[0]) if config.MODEL.ARGS else net(ckpt_dir=ckpt_dir)

    logger.info(f"Model dropout: {model.dropout_d}")
    model = model.to(device)
    optimizer = None
    scheduler = None

    if config.MODE == 'train':
        optimizer = build_optimizer(config, model, logger)

    config.defrost()
    config.TRAIN.START_EPOCH = model.load(device, logger, optimizer, best=config.MODE != 'train' or not config.TRAIN.RESUME_LAST)
    config.freeze()

    if config.MODE == 'train' and len(config.MODEL.FINE_TUNE) > 0:
        for param in model.parameters():
            param.requires_grad = False
        for layer in config.MODEL.FINE_TUNE:
            logger.info(f"Fine-tuning layer: {layer}")
            getattr(model, layer).requires_grad_(True)
            getattr(model, layer).reset_parameters()

    model.show_parameter_number(logger)

    if config.MODE == 'train':
        if len(config.TRAIN.LR_SCHEDULER.NAME) > 0:
            scheduler_args = config.TRAIN.LR_SCHEDULER.ARGS[0]
            scheduler_args.setdefault('last_epoch', config.TRAIN.START_EPOCH - 1)
            scheduler = getattr(lr_scheduler, config.TRAIN.LR_SCHEDULER.NAME)(optimizer=optimizer, **scheduler_args)
            logger.info(f"Using scheduler: {config.TRAIN.LR_SCHEDULER.NAME} with args: {scheduler_args}")
        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            from apex import amp
            logger.info(f"Using AMP (opt level: {config.AMP_OPT_LEVEL})")
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL, verbosity=0)
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.TRAIN.DEVICE])

    criterion = build_criterion(config, logger)
    if optimizer:
        logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']}")

    return model, optimizer, criterion, scheduler
