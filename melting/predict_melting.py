import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch

from torch.utils.data import DataLoader, DistributedSampler
import wandb

from neuralop import get_model
from training.trainer import Trainer
from losses.mask_data_losses import LpLoss
from utils.load_data import load_melting_dataset
from utils.plot_animation import plot_channel_animation,compare_tensors_animation
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.utils import get_project_root


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./melting_config.yaml", config_name="default", config_folder="./configs"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key(api_key_file="configs/wandb_api_key.txt"))
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.tfno3d.n_layers,
                config.tfno3d.hidden_channels,
                config.tfno3d.n_modes_height,
                config.tfno3d.n_modes_width,
                config.tfno3d.n_modes_depth,
                config.tfno3d.factorization,
                config.tfno3d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_args =  dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

data_folder = get_project_root() / config.data.folder
# Loading the Darcy flow dataset
train_loader, test_loaders, data_processor = load_melting_dataset(data_root = data_folder,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
)

model = get_model(config)
model = model.from_checkpoint(save_folder="./ckpt", save_name="model")

# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(model=model,
                                             in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels,
                                             use_distributed=config.distributed.use_distributed,
                                             device=device)

# Reconfigure DataLoaders to use a DistributedSampler 
# if in distributed data parallel mode
if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(dataset=train_db,
                              batch_size=config.data.batch_size,
                              sampler=train_sampler)
    for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
        
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(dataset=test_db,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=test_sampler)

# Creating the losses
#l2loss = LpLoss(d=3, p=2, data_processor = data_processor,loss_type=config.opt.loss_type, mask_channel_outputs=[0,2,3], mask_channel=5)
l2loss = LpLoss(d=3, p=2, data_processor=data_processor,loss_type=config.opt.loss_type, mask_channel_outputs=None, mask_channel=5)

eval_losses = {"l2": l2loss}

model.eval()
data_processor.eval()
with torch.no_grad():
    for loader_name, test_loader in test_loaders.items():
        n_samples = 0
        for idx, sample in enumerate(test_loader):
            if data_processor is not None:
                sample = data_processor.preprocess(sample)
            else:
                # load data to device if no preprocessor exists
                sample = {
                    k: v.to(device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }

            n_samples += sample["y"].size(0)

            out = model(**sample)

            print(data_processor.training)
            if data_processor is not None:
                out, sample = data_processor.postprocess(out, sample)
            
            eval_step_losses = {}

            for loss_name, loss in eval_losses.items():
                val_loss, val_loss_channel = loss(out, **sample)
                eval_step_losses[loss_name+"_channel"] = val_loss_channel/n_samples
                eval_step_losses[loss_name] = val_loss/n_samples

            print(out.shape)
            print(eval_step_losses)
            #plot_channel_animation(out)
            compare_tensors_animation(out,sample["y"])
            #break

