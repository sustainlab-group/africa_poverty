from utils.run import get_full_experiment_name, make_log_and_ckpt_dirs

import os
import subprocess
import time


root = '/atlas/u/chrisyeh/africa_poverty/'

default_params = {
    'ckpt_dir': os.path.join(root, 'ckpts/DHSIncountry/'),
    'log_dir': os.path.join(root, 'logs/DHSIncountry/'),
    'label_name': 'wealthpooled',
    'batcher': 'base',
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

# for OOC keep_frac training
# HPARAMS = [
#     ('ms', 'None', 'incountryA', 'samescaled', 1e-1, 1e-4),
#     ('ms', 'None', 'incountryB', 'samescaled', 1e-1, 1e-4),
#     ('ms', 'None', 'incountryC', 'samescaled', 1e-0, 1e-4),
#     ('ms', 'None', 'incountryD', 'samescaled', 1e-3, 1e-4),
#     ('ms', 'None', 'incountryE', 'samescaled', 1e-3, 1e-4),
#     ('ms', 'split', 'incountryA', 'samescaled', 1e-1, 1e-2),
#     ('ms', 'split', 'incountryB', 'samescaled', 1e-3, 1e-2),
#     ('ms', 'split', 'incountryC', 'samescaled', 1e-1, 1e-4),
#     ('ms', 'split', 'incountryD', 'samescaled', 1e-0, 1e-2),
#     ('ms', 'split', 'incountryE', 'samescaled', 1e-1, 1e-4),
#     ('None', 'split', 'incountryA', 'random', 1e-1, 1e-4),
#     ('None', 'split', 'incountryB', 'random', 1e-1, 1e-4),
#     ('None', 'split', 'incountryC', 'random', 1e-0, 1e-4),
#     ('None', 'split', 'incountryD', 'random', 1e-0, 1e-2),
#     ('None', 'split', 'incountryE', 'random', 1e-0, 1e-4),
#     ('rgb', 'None', 'incountryA', 'same', 1e-3, 1e-2),
#     ('rgb', 'None', 'incountryB', 'same', 1e-1, 1e-4),
#     ('rgb', 'None', 'incountryC', 'same', 1e-3, 1e-4),
#     ('rgb', 'None', 'incountryD', 'same', 1e-3, 1e-5),
#     ('rgb', 'None', 'incountryE', 'same', 1e-3, 1e-4),
#     ('rgb', 'split', 'incountryA', 'samescaled', 1e-1, 1e-4),
#     ('rgb', 'split', 'incountryB', 'samescaled', 1e-1, 1e-2),
#     ('rgb', 'split', 'incountryC', 'samescaled', 1e-3, 1e-2),
#     ('rgb', 'split', 'incountryD', 'samescaled', 1e-3, 1e-2),
#     ('rgb', 'split', 'incountryE', 'samescaled', 1e-3, 1e-2),
# ]

# for incountry training
# (ls_bands, nl_band, dataset, hs_weight_init)
HPARAMS = [
    ('ms', 'None', 'incountryA', 'samescaled'),
    ('ms', 'None', 'incountryB', 'samescaled'),
    ('ms', 'None', 'incountryC', 'samescaled'),
    ('ms', 'None', 'incountryD', 'samescaled'),
    ('ms', 'None', 'incountryE', 'samescaled'),
#     ('ms', 'split', 'incountryA', 'samescaled'),
#     ('ms', 'split', 'incountryB', 'samescaled'),
#     ('ms', 'split', 'incountryC', 'samescaled'),
#     ('ms', 'split', 'incountryD', 'samescaled'),
#     ('ms', 'split', 'incountryE', 'samescaled'),
    ('None', 'split', 'incountryA', 'random'),
    ('None', 'split', 'incountryB', 'random'),
    ('None', 'split', 'incountryC', 'random'),
    ('None', 'split', 'incountryD', 'random'),
    ('None', 'split', 'incountryE', 'random'),
#     ('rgb', 'None', 'incountryA', 'same'),
#     ('rgb', 'None', 'incountryB', 'same'),
#     ('rgb', 'None', 'incountryC', 'same'),
#     ('rgb', 'None', 'incountryD', 'same'),
#     ('rgb', 'None', 'incountryE', 'same'),
#     ('rgb', 'split', 'incountryA', 'samescaled'),
#     ('rgb', 'split', 'incountryB', 'samescaled'),
#     ('rgb', 'split', 'incountryC', 'samescaled'),
#     ('rgb', 'split', 'incountryD', 'samescaled'),
#     ('rgb', 'split', 'incountryE', 'samescaled'),
]

PYTHON_COMMAND_TEMPLATE = 'python train_directly.py \
    --label_name wealthpooled \
    --batcher "{batcher}" \
    --model_name resnet --num_layers 18 \
    --lr_decay 0.96 \
    --batch_size 64 \
    --augment=True \
    --max_epochs 150 --eval_every 1 --print_every 40 \
    --ooc=False \
    --keep_frac "{keep_frac}" \
    --gpu 0 --num_threads 5 \
    --ckpt_dir {ckpt_dir} \
    --log_dir {log_dir} \
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
    ls_bands, nl_band, dataset, hs_weight_init, reg, lr, keep_frac, seed = hparams_tup
    experiment_name = '{dataset}_18preact_{bands}_{init}'.format(
        dataset=dataset,
        bands=get_bandname(ls_bands, nl_band),
        init=hs_weight_init)

    imagenet_weights_path = 'None'
    if hs_weight_init in ['same', 'samescaled']:
        imagenet_weights_path = '/atlas/group/model_weights/imagenet_resnet18_tensorpack.npz'

    command = PYTHON_COMMAND_TEMPLATE.format(
        ckpt_dir=default_params['ckpt_dir'],
        log_dir=default_params['log_dir'],
        batcher=default_params['batcher'],
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
# - DHS incountryA
#   - MS: 43 GB RAM, 5439MiB GPU
#   - NL: 13 GB RAM, 4415MiB GPU


def get_mem(bands):
    return {
        'rgb': 20,
        'rgbnl': 32,
        'ms': 43,
        'msnl': 56,
        'nl': 15,
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
    mem = int(get_mem(bandname) * keep_frac + 3)
    return mem


# order of params: ls_bands, nl_band, dataset, hs_weight_init, reg, lr, keep_frac, seed
all_hparams = []
for hparams_tup in HPARAMS:
    for reg in [1e-0, 1e-1, 1e-2, 1e-3]:
        for lr in [1e-2, 1e-3, 1e-4]:
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
