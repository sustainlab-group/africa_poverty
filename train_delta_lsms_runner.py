from utils.run import get_full_experiment_name, make_log_and_ckpt_dirs

import os
import subprocess
import time


root = '/atlas/u/chrisyeh/africa_poverty/'
FOLDS = ['A', 'B', 'C', 'D', 'E']

# incountry delta regression
NAME = 'LSMSDeltaIncountry'
default_params = {
    'ckpt_dir': os.path.join(root, f'ckpts/{NAME}/'),
    'log_dir': os.path.join(root, f'logs/{NAME}/'),
    'label_name': 'None',
    'batcher': 'delta',
    'orig_labels': False,
    'augment': 'bidir',
    'weighted': True
}

# incountry delta regression w/ orig_labels
# NAME = 'LSMSDeltaIncountry_origlabels'
# default_params = {
#     'ckpt_dir': os.path.join(root, f'ckpts/{NAME}/'),
#     'log_dir': os.path.join(root, f'logs/{NAME}/'),
#     'label_name': 'wealthpooled',
#     'batcher': 'delta',
#     'orig_labels': True,
#     'augment': 'bidir',
#     'weighted': False
# }

# incountry delta regression w/ orig_labels and only forward direction
# NAME = 'LSMSDeltaIncountry_origlabels'
# default_params = {
#     'ckpt_dir': os.path.join(root, f'ckpts/{NAME}/'),
#     'log_dir': os.path.join(root, f'logs/{NAME}/'),
#     'label_name': 'wealthpooled',
#     'batcher': 'delta',
#     'orig_labels': True,
#     'augment': 'forward',
#     'weighted': False
# }

# incountry index of delta regression
# NAME = 'LSMSIndexOfDeltaIncountry'
# default_params = {
#     'ckpt_dir': os.path.join(root, f'ckpts/{NAME}/'),
#     'log_dir': os.path.join(root, f'logs/{NAME}/'),
#     'label_name': 'None',
#     'batcher': 'delta',
#     'orig_labels': False,
#     'augment': 'bidir',
#     'weighted': True
# }

# incountry delta classification
# NAME = 'LSMSDeltaClassIncountry'
# default_params = {
#     'ckpt_dir': os.path.join(root, f'ckpts/{NAME}/'),
#     'log_dir': os.path.join(root, f'logs/{NAME}/'),
#     'label_name': 'wealthpooled',
#     'batcher': 'deltaclass',
#     'orig_labels': False,
#     'augment': 'bidir',
#     'weighted': False
# }

# (ls_bands, nl_band, dataset, hs_weight_init)
HPARAMS = [('ms', 'None', f'{NAME}{f}', 'random') for f in FOLDS]  # ms
HPARAMS += [('ms', 'split', f'{NAME}{f}', 'random') for f in FOLDS]  # msnl
HPARAMS += [('None', 'split', f'{NAME}{f}', 'random') for f in FOLDS]  # nl
# HPARAMS += [('rgb', 'None', f'{NAME}{f}', 'random') for f in FOLDS]  # rgb
# HPARAMS += [('rgb', 'split', f'{NAME}{f}', 'random') for f in FOLDS]  # rgbnl

# order of params: (ls_bands, nl_band, dataset, hs_weight_init, reg, lr, keep_frac, seed)
all_hparams = []
for hparams_tup in HPARAMS:
    for reg in [1e-1, 1e-2, 1e-3]:  # [1e-0, 1e-1, 1e-2, 1e-3]:
        for lr in [1e-4]:  # [1e-2, 1e-3, 1e-4, 1e-5]:
            for keep in [1.0]:  # [0.05, 0.10, 0.25, 0.5]:
                for seed in [123]:  # [123, 456, 789]:
                    new_tup = tuple(list(hparams_tup) + [reg, lr, keep, seed])
                    all_hparams.append(new_tup)

# sort hparams by ls_bands
all_hparams = sorted(all_hparams, key=lambda x: x[0])


PYTHON_COMMAND_TEMPLATE = 'python train_delta.py \
    --model_name resnet --num_layers 18 \
    --max_epochs 150 --eval_every 1 --print_every 40 \
    --lr_decay 0.96 \
    --batch_size 64 \
    --gpu 0 --num_threads 5 \
    --cache train,train_eval,val \
    --ooc=False \
    \
    --label_name "{label_name}" \
    --orig_labels="{orig_labels}" \
    --weighted="{weighted}" \
    --batcher "{batcher}" \
    --augment "{augment}" \
    --ckpt_dir "{ckpt_dir}" \
    --log_dir "{log_dir}" \
    --keep_frac "{keep_frac}" \
    --seed "{seed}" \
    --experiment_name "{experiment_name}" \
    --dataset "{dataset}" \
    --ls_bands "{ls_bands}" --nl_band "{nl_band}" \
    --lr "{lr}" --fc_reg "{reg}" --conv_reg "{reg}" \
    --imagenet_weights_path "{imagenet_weights_path}" \
    --hs_weight_init "{hs_weight_init}"'


def hparams_to_command(hparams_tup):
    '''
    Args
    - hparams_tup: tuple, (ls_bands, nl_band, dataset, hs_weight_init, reg, lr, keep_frac, seed)

    Returns
    - command: str, invocation of 'python train_delta.py ...'
    '''
    ls_bands, nl_band, dataset, hs_weight_init, reg, lr, keep_frac, seed = hparams_tup
    if default_params['orig_labels']:
        origlabels = 'origlabels_'
    else:
        origlabels = ''
    experiment_name = '{dataset}_bidir_{origlabels}18preact_{bands}_{init}'.format(
        dataset=dataset,
        origlabels=origlabels,
        bands=get_bandname(ls_bands, nl_band),
        init=hs_weight_init)

    imagenet_weights_path = 'None'
    if hs_weight_init in ['same', 'samescaled']:
        imagenet_weights_path = '/atlas/group/model_weights/imagenet_resnet18_tensorpack.npz'

    command = PYTHON_COMMAND_TEMPLATE.format(
        label_name=default_params['label_name'],
        orig_labels=default_params['orig_labels'],
        weighted=default_params['weighted'],
        batcher=default_params['batcher'],
        augment=default_params['augment'],
        ckpt_dir=default_params['ckpt_dir'],
        log_dir=default_params['log_dir'],
        keep_frac=keep_frac,
        seed=seed,
        experiment_name=experiment_name,
        dataset=dataset,
        ls_bands=ls_bands,
        nl_band=nl_band,
        lr=lr,
        reg=reg,
        imagenet_weights_path=imagenet_weights_path,
        hs_weight_init=hs_weight_init)

    full_experiment_name = get_full_experiment_name(
        experiment_name, 64, reg, reg, lr)
    log_dir, _ = make_log_and_ckpt_dirs(
        default_params['log_dir'], default_params['ckpt_dir'], full_experiment_name)

    output_path = os.path.join(log_dir, 'output.log')
    command = f'{command} >& "{output_path}" &'
    return command


# Empirical memory requirements:
# - LSMSDeltaIncountry (weighted)
#   - MS: ??
# - LSMSDeltaIncountry_origlabels
#   - MS: 12 GB RAM, 5517 MiB GPU
# - LSMSIndexOfDeltaIncountry
#   - MS: 11 GB RAM, 5517 MiB GPU
#   - NL: 5 GB RAM, 4483 MiB GPU


def get_mem(bands: str) -> int:
    return {
        'rgb': 10,
        'rgbnl': 11,
        'ms': 12,
        'msnl': 14,
        'nl': 10,
    }[bands]


def get_bandname(ls_bands: str, nl_band: str):
    return {
        ('ms', 'None'): 'ms',
        ('ms', 'split'): 'msnl',
        ('None', 'split'): 'nl',
        ('rgb', 'None'): 'rgb',
        ('rgb', 'split'): 'rgbnl',
    }[(ls_bands, nl_band)]


def get_mem_for_hparams(hparams):
    bandname = get_bandname(hparams[0], hparams[1])
    keep_frac = hparams[-2]
    min_mem = 10
    buffer = 3
    mem = int(get_mem(bandname) * keep_frac + buffer)
    mem = max(min_mem, mem)
    return mem


outputs_dir = os.path.join(root, 'outputs', str(int(time.time())))
os.makedirs(outputs_dir, exist_ok=True)

for i in range(0, len(all_hparams), 2):
    print(i)
    hparams1 = all_hparams[i]
    command1 = hparams_to_command(hparams1)
    mem1 = get_mem_for_hparams(hparams1)

    command2 = ''
    mem2 = 0
    if i + 1 < len(all_hparams):
        hparams2 = all_hparams[i + 1]
        command2 = hparams_to_command(hparams2)
        mem2 = get_mem_for_hparams(hparams2)

    command = command1 + '\n\n' + command2
    mem = mem1 + mem2

    with open(os.path.join(root, 'scripts', 'train_model_slurm.sh'), 'r') as template_file:
        template = template_file.read()
        full_slurm_script = template.format(
            SLURM_MEM=f'{mem}G',
            SLURM_JOB_NAME=f'slurm_{i}',
            SLURM_OUTPUT_LOG=os.path.join(outputs_dir, f'slurm_{i}.log'),
            content=command)

        slurm_sh_path = os.path.join(outputs_dir, f'slurm_{i}.sh')
        with open(slurm_sh_path, 'w') as f:
            f.write(full_slurm_script)
        subprocess.run(['sbatch', slurm_sh_path])


# 'experiment_name': 'direct_2009-17_18preact_nl_keep{}_seed{}'.format(
#     str(keep_frac).lstrip('0'), seed),
#
# 'init_ckpt_dir': '/atlas/u/chrisyeh/hyperspectral_Resnet/ckpts/transfer_2009-17nl_nlcenter_18preact_rgb_b64_fc001_conv001_lr0001/',
# 'init_ckpt_dir': '/atlas/u/chrisyeh/hyperspectral_Resnet/ckpts/transfer_2009-17nl_nlcenter_18preact_ms_b64_fc001_conv001_lr0001/',
# 'exclude_final_layer': True,
