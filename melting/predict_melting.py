import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch

from torch.utils.data import DataLoader, DistributedSampler
import wandb

from neuralop import get_model
from training.trainer import Trainer
from losses.mask_data_losses import LpLoss
from utils.load_data import load_melting_dataset
from utils.plot_animation import plot_channel_animation,compare_tensors_animation,compare_batches_animation,compare_batches_velocity_animation
from utils.melting_formula import calculate_alpha
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
config.distributed.use_distributed = False

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

train_data_folder = get_project_root() / config.data.train_data_folder
test_data_folder = get_project_root() / config.pred_data.test_data_folder
# Loading the Darcy flow dataset
train_loader, test_loaders, data_processor = load_melting_dataset(train_data_root = train_data_folder,
    test_data_root = test_data_folder,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.pred_data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.pred_data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
)

print(data_processor.in_normalizer.min_val.squeeze())
print(data_processor.in_normalizer.max_val.squeeze())

print(data_processor.out_normalizer.min_val.squeeze())
print(data_processor.out_normalizer.max_val.squeeze())

model = get_model(config)
model.load_checkpoint(save_folder=config.tfno3d.save_dir, save_name="model")

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
            sample0 = sample["x"].clone()
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
            mask = sample0[:,5:6,...]
            out_prgh = out[:,0:1,...]*mask
            out_T = out[:,1:2,...]
            out_Ux = out[:,2:3,...]*mask
            out_Uy = out[:,3:4,...]*mask
            out_Umag = torch.sqrt(out_Ux*out_Ux + out_Uy*out_Uy)

            out_alpha = mask*calculate_alpha(out_T,T_l=303.43,T_s=302.43)

            gt = sample["y"].clone()
            gt_prgh = sample["y"][:,0:1,...]*mask
            gt_T = sample["y"][:,1:2,...]
            gt_Ux = sample["y"][:,2:3,...]*mask
            gt_Uy = sample["y"][:,3:4,...]*mask
            gt_Umag = torch.sqrt(gt_Ux*gt_Ux + gt_Uy*gt_Uy)
            gt_alpha = mask*calculate_alpha(gt_T,T_l=303.43,T_s=302.43)
            #gt[:,2:,...] = sample["y"][:,2:,...]*mask
            #compare_tensors_animation(out,gt,batch_num=30)
            #prefix="regular_time10_res50"
            prefix="newdata_case_test_res100_long4000"
            compare_batches_animation(out_prgh,gt_prgh,sample_num=idx,channel_names={0:"prgh"},prefix=prefix)
            compare_batches_animation(out_T,gt_T,sample_num=idx,channel_names={0:"T"},prefix=prefix)
            compare_batches_animation(out_Ux,gt_Ux,sample_num=idx,channel_names={0:"Ux"},prefix=prefix)
            compare_batches_animation(out_Uy,gt_Uy,sample_num=idx,channel_names={0:"Uy"},prefix=prefix)
            compare_batches_animation(out_Umag,gt_Umag,sample_num=idx,channel_names={0:"Umag"},prefix=prefix)
            compare_batches_animation(out_alpha,gt_alpha,sample_num=idx,channel_names={0:"alpha"},prefix=prefix)
            compare_batches_velocity_animation(out_Ux,out_Uy,gt_Ux,gt_Uy,sample_num=idx,downsample_factor=1,arrow_scale=25,prefix=prefix)
            break

