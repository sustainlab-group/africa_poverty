from utils.run import get_full_experiment_name, make_log_and_ckpt_dirs

import os
import subprocess
import time


root = '/atlas/u/chrisyeh/africa_poverty/'

default_params = {
    'ckpt_dir': os.path.join(root, 'ckpts/'),
    'log_dir': os.path.join(root, 'logs/'),
    'label_name': 'wealthpooled',
    'batcher': 'deltaclass',
    'orig_labels': False,
    'batch_size': 64,
    'model_name': 'resnet',
    'num_layers': 18,
    'augment': True,
    'lr_decay': 0.96,
    'num_threads': 5,
    'max_epochs': 200,
    'eval_every': 1,
    'print_every': 40,
    'cache': ['train', 'train_eval', 'val'],
    'gpu': 0,
    'gpu_usage': 0.96
}

# for incountry delta regression
# HPARAMS = [
#     ('ms', 'None', 'LSMSDeltaIncountryA', 'random'),  # 'samescaled'
#     ('ms', 'None', 'LSMSDeltaIncountryB', 'random'),  # 'samescaled'
#     ('ms', 'None', 'LSMSDeltaIncountryC', 'random'),  # 'samescaled'
#     ('ms', 'None', 'LSMSDeltaIncountryD', 'random'),  # 'samescaled'
#     ('ms', 'None', 'LSMSDeltaIncountryE', 'random'),  # 'samescaled'
#     ('ms', 'split', 'LSMSDeltaIncountryA', 'random'),  # 'samescaled'
#     ('ms', 'split', 'LSMSDeltaIncountryB', 'random'),  # 'samescaled'
#     ('ms', 'split', 'LSMSDeltaIncountryC', 'random'),  # 'samescaled'
#     ('ms', 'split', 'LSMSDeltaIncountryD', 'random'),  # 'samescaled'
#     ('ms', 'split', 'LSMSDeltaIncountryE', 'random'),  # 'samescaled'
#     ('None', 'split', 'LSMSDeltaIncountryA', 'random'),
#     ('None', 'split', 'LSMSDeltaIncountryB', 'random'),
#     ('None', 'split', 'LSMSDeltaIncountryC', 'random'),
#     ('None', 'split', 'LSMSDeltaIncountryD', 'random'),
#     ('None', 'split', 'LSMSDeltaIncountryE', 'random'),
#     ('rgb', 'None', 'LSMSDeltaIncountryA', 'same'),
#     ('rgb', 'None', 'LSMSDeltaIncountryB', 'same'),
#     ('rgb', 'None', 'LSMSDeltaIncountryC', 'same'),
#     ('rgb', 'None', 'LSMSDeltaIncountryD', 'same'),
#     ('rgb', 'None', 'LSMSDeltaIncountryE', 'same'),
#     ('rgb', 'split', 'LSMSDeltaIncountryA', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaIncountryB', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaIncountryC', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaIncountryD', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaIncountryE', 'samescaled'),
# ]

# for incountry delta classification
HPARAMS = [
    ('ms', 'None', 'LSMSDeltaClassIncountryA', 'random'),  # 'samescaled'
    ('ms', 'None', 'LSMSDeltaClassIncountryB', 'random'),  # 'samescaled'
    ('ms', 'None', 'LSMSDeltaClassIncountryC', 'random'),  # 'samescaled'
    ('ms', 'None', 'LSMSDeltaClassIncountryD', 'random'),  # 'samescaled'
    ('ms', 'None', 'LSMSDeltaClassIncountryE', 'random'),  # 'samescaled'
    ('ms', 'split', 'LSMSDeltaClassIncountryA', 'random'),  # 'samescaled'
    ('ms', 'split', 'LSMSDeltaClassIncountryB', 'random'),  # 'samescaled'
    ('ms', 'split', 'LSMSDeltaClassIncountryC', 'random'),  # 'samescaled'
    ('ms', 'split', 'LSMSDeltaClassIncountryD', 'random'),  # 'samescaled'
    ('ms', 'split', 'LSMSDeltaClassIncountryE', 'random'),  # 'samescaled'
#     ('None', 'split', 'LSMSDeltaClassIncountryA', 'random'),
#     ('None', 'split', 'LSMSDeltaClassIncountryB', 'random'),
#     ('None', 'split', 'LSMSDeltaClassIncountryC', 'random'),
#     ('None', 'split', 'LSMSDeltaClassIncountryD', 'random'),
#     ('None', 'split', 'LSMSDeltaClassIncountryE', 'random'),
#     ('rgb', 'None', 'LSMSDeltaClassIncountryA', 'same'),
#     ('rgb', 'None', 'LSMSDeltaClassIncountryB', 'same'),
#     ('rgb', 'None', 'LSMSDeltaClassIncountryC', 'same'),
#     ('rgb', 'None', 'LSMSDeltaClassIncountryD', 'same'),
#     ('rgb', 'None', 'LSMSDeltaClassIncountryE', 'same'),
#     ('rgb', 'split', 'LSMSDeltaClassIncountryA', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaClassIncountryB', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaClassIncountryC', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaClassIncountryD', 'samescaled'),
#     ('rgb', 'split', 'LSMSDeltaClassIncountryE', 'samescaled'),
]

PYTHON_COMMAND_TEMPLATE = 'python train_delta.py \
    --label_name wealthpooled \
    --orig_labels={orig_labels} \
    --batcher {batcher} \
    --model_name resnet --num_layers 18 \
    --lr_decay 0.96 \
    --batch_size 64 \
    --augment=True \
    --max_epochs 200 --eval_every 1 --print_every 40 \
    --ooc=False \
    --keep_frac "{keep_frac}" \
    --gpu 0 --gpu_usage 0.96 --num_threads 5 \
    --ckpt_dir /atlas/u/chrisyeh/africa_poverty/ckpts/ \
    --log_dir /atlas/u/chrisyeh/africa_poverty/logs/ \
    --cache train,train_eval,val \
    --seed "{seed}" \
    \
    --experiment_name "{experiment_name}" \
    --dataset "{dataset}" \
    --ls_bands "{ls_bands}" \
    --nl_band "{nl_band}" \
    --lr "{lr}" \
    --fc_reg "{reg}" --conv_reg "{reg}" \
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
    experiment_name = '{dataset}_origlabels_18preact_{bands}_{init}'.format(
        dataset=dataset,
        bands=get_bandname(ls_bands, nl_band),
        init=hs_weight_init)

    imagenet_weights_path = 'None'
    if hs_weight_init in ['same', 'samescaled']:
        imagenet_weights_path = '/atlas/group/model_weights/imagenet_resnet18_tensorpack.npz'

    command = PYTHON_COMMAND_TEMPLATE.format(
        batcher=default_params['batcher'],
        orig_labels=default_params['orig_labels'],
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
        experiment_name, default_params['batch_size'], reg, reg, lr)
    log_dir, _ = make_log_and_ckpt_dirs(
        default_params['log_dir'], default_params['ckpt_dir'], full_experiment_name)

    output_path = os.path.join(log_dir, 'output.log')
    command = f'{command} >& "{output_path}" &'
    return command


# Empirical memory requirements:
# - delta-origlabels
#   - MS: 12 GB RAM, 5517 MiB GPU
#   - MSNL: ?, ?


def get_mem(bands):
    return {
        'rgb': 10,
        'rgbnl': 11,
        'ms': 12,
        'msnl': 14,
        'nl': 10,
    }[bands]


def get_bandname(ls_bands, nl_band):
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
    buffer = 4
    mem = int(get_mem(bandname) * keep_frac + buffer)
    mem = max(min_mem, mem)
    return mem


# order of params: ls_bands, nl_band, dataset, hs_weight_init, reg, lr, keep_frac, seed
all_hparams = []
for hparams_tup in HPARAMS:
    for reg in [1e-0, 1e-1, 1e-2, 1e-3]:
        for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
            for keep in [1.0]:  # [0.05, 0.10, 0.25, 0.5]:
                for seed in [123]:  # [123, 456, 789]:
                    new_tup = tuple(list(hparams_tup) + [reg, lr, keep, seed])
                    all_hparams.append(new_tup)

# sort hparams by ls_bands
all_hparams = sorted(all_hparams, key=lambda x: x[0])

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
