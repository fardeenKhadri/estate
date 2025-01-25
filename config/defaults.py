import os
import logging
from yacs.config import CfgNode as CN

# Base configuration object
_C = CN()

# General settings
_C.DEBUG = False
_C.MODE = 'train'
_C.VAL_NAME = 'val'
_C.TAG = 'default'
_C.COMMENT = 'Add descriptive comments here'
_C.SHOW_BAR = True
_C.SAVE_EVAL = False

# Model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'LGT_Net'  # Default model name
_C.MODEL.SAVE_BEST = True
_C.MODEL.SAVE_LAST = True
_C.MODEL.ARGS = [{
    "decoder_name": "Transformer",  # Options: 'Transformer', 'LSTM'
    "output_name": "LGT",          # Options: 'LGT', 'LED', 'Horizon'
    "backbone": "resnet50",        # Backbone model for feature extraction
    "dropout": 0.1,                # Dropout rate
    "win_size": 8,                 # Window size for Transformer
    "depth": 6                     # Depth of Transformer layers
}]
_C.MODEL.FINE_TUNE = []

# Training settings
_C.TRAIN = CN()
_C.TRAIN.SCRATCH = False
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.DETERMINISTIC = False
_C.TRAIN.SAVE_FREQ = 5
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.RESUME_LAST = True
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.DEVICE = 'cuda'

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = ''
_C.TRAIN.LR_SCHEDULER.ARGS = []

# Optimizer settings
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Criterion settings
_C.TRAIN.CRITERION = CN()
_C.TRAIN.CRITERION.BOUNDARY = CN()
_C.TRAIN.CRITERION.BOUNDARY.NAME = 'boundary'
_C.TRAIN.CRITERION.BOUNDARY.LOSS = 'BoundaryLoss'
_C.TRAIN.CRITERION.BOUNDARY.WEIGHT = 0.0
_C.TRAIN.CRITERION.BOUNDARY.WEIGHTS = []
_C.TRAIN.CRITERION.BOUNDARY.NEED_ALL = True

# Checkpoint settings
_C.CKPT = CN()
_C.CKPT.PYTORCH = './'
_C.CKPT.ROOT = "./checkpoints"
_C.CKPT.DIR = os.path.join(_C.CKPT.ROOT, _C.MODEL.NAME, _C.TAG)
_C.CKPT.RESULT_DIR = os.path.join(_C.CKPT.DIR, 'results', _C.MODE)

# Logger settings
_C.LOGGER = CN()
_C.LOGGER.DIR = os.path.join(_C.CKPT.DIR, "logs")
_C.LOGGER.LEVEL = logging.DEBUG

# Mixed precision optimization
_C.AMP_OPT_LEVEL = 'O1'

# Miscellaneous settings
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False

# Distributed training settings
_C.LOCAL_RANK = 0
_C.WORLD_SIZE = 0

# Data settings
_C.DATA = CN()
_C.DATA.DATASET = 'mp3d'
_C.DATA.DIR = ''
_C.DATA.BATCH_SIZE = 8
_C.DATA.NUM_WORKERS = 8

# Evaluation settings
_C.EVAL = CN()
_C.EVAL.POST_PROCESSING = None
_C.EVAL.NEED_CPE = False
_C.EVAL.NEED_F1 = False
_C.EVAL.NEED_RMSE = False
_C.EVAL.FORCE_CUBE = False

# Functions to manipulate configuration
def merge_from_file(cfg_path):
    config = _C.clone()
    config.merge_from_file(cfg_path)
    return config

def get_config(args=None):
    config = _C.clone()
    if args:
        if 'cfg' in args and args.cfg:
            config.merge_from_file(args.cfg)
        if 'mode' in args and args.mode:
            config.MODE = args.mode
        if 'debug' in args and args.debug:
            config.DEBUG = args.debug
        if 'hidden_bar' in args and args.hidden_bar:
            config.SHOW_BAR = False
        if 'bs' in args and args.bs:
            config.DATA.BATCH_SIZE = args.bs
        if 'save_eval' in args and args.save_eval:
            config.SAVE_EVAL = True
        if 'val_name' in args and args.val_name:
            config.VAL_NAME = args.val_name
        if 'post_processing' in args and args.post_processing:
            config.EVAL.POST_PROCESSING = args.post_processing
        if 'need_cpe' in args and args.need_cpe:
            config.EVAL.NEED_CPE = args.need_cpe
        if 'need_f1' in args and args.need_f1:
            config.EVAL.NEED_F1 = args.need_f1
        if 'need_rmse' in args and args.need_rmse:
            config.EVAL.NEED_RMSE = args.need_rmse
        if 'force_cube' in args and args.force_cube:
            config.EVAL.FORCE_CUBE = args.force_cube
        if 'wall_num' in args and args.wall_num:
            config.DATA.WALL_NUM = args.wall_num

    if config.MODEL.ARGS:
        args = config.MODEL.ARGS[0]
        config.CKPT.DIR = os.path.join(config.CKPT.ROOT, f"{args['decoder_name']}_{args['output_name']}_Net",
                                       config.TAG, 'debug' if config.DEBUG else '')
        config.CKPT.RESULT_DIR = os.path.join(config.CKPT.DIR, 'results', config.MODE)
        config.LOGGER.DIR = os.path.join(config.CKPT.DIR, "logs")

    config.freeze()
    return config

def get_rank_config(cfg, local_rank, world_size):
    local_rank = 0 if local_rank is None else local_rank
    config = cfg.clone()
    config.defrost()
    if world_size > 1:
        ids = config.TRAIN.DEVICE.split(':')[-1].split(',') if ':' in config.TRAIN.DEVICE else range(world_size)
        config.TRAIN.DEVICE = f'cuda:{ids[local_rank]}'

    config.LOCAL_RANK = local_rank
    config.WORLD_SIZE = world_size
    config.SEED = config.SEED + local_rank

    config.freeze()
    return config
