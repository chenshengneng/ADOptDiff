import argparse
import os

import torch

from src import utils
from src.model_multi import DDPM
from src.visualizer import save_xyz_file, save_xyz_file_fa
from src.datasets import collate_mr, MultiRDataset_anchor
from tqdm import tqdm

import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--samples', action='store', type=str, required=True)
parser.add_argument('--data', action='store', type=str, required=False, default=None)
parser.add_argument('--prefix', action='store', type=str, required=True)
parser.add_argument('--n_samples', action='store', type=int, required=True)
parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
parser.add_argument('--device', action='store', type=str, required=True)
parser.add_argument('--rgroup_size_model', action='store', type=str, required=False, default=None)
args = parser.parse_args()

experiment_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

output_dir = os.path.join(args.samples, experiment_name)

os.makedirs(output_dir, exist_ok=True)

def check_if_generated(_output_dir, _uuids, n_samples):
    generated = True
    starting_points = []
    for _uuid in _uuids:
        uuid_dir = os.path.join(_output_dir, _uuid)
        numbers = []
        for fname in os.listdir(uuid_dir):
            try:
                num = int(fname.split('_')[0])
                numbers.append(num)
            except:
                continue
        if len(numbers) == 0 or max(numbers) != n_samples - 1:
            generated = False
            if len(numbers) == 0:
                starting_points.append(0)
            else:
                starting_points.append(max(numbers) - 1)

    if len(starting_points) > 0:
        starting = min(starting_points)
    else:
        starting = None

    return generated, starting


collate_fn = collate_mr
sample_fn = None

# Loading model form checkpoint (all hparams will be automatically set)
model = DDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)

# Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
model.val_data_prefix = args.prefix

# In case <Anonymous> will run my model or vice versa
if args.data is not None:
    model.data_path = args.data

# Less sampling steps
if args.n_steps is not None:
    model.edm.T = args.n_steps

# Setting up the model
model = model.eval().to(args.device)
model.setup(stage='val')

model.batch_size = 1
# Getting the dataloader
dataloader = model.val_dataloader(collate_fn=collate_fn)
print(f'Dataloader contains {len(dataloader)} batches')

center_of_mass_list = []

time_start = time.time()
for batch_idx, data in enumerate(dataloader):
    uuids = []
    true_names = []
    scaf_names = []
    pock_names = []
    for uuid in data['uuid']:
        uuid = str(uuid)
        uuids.append(uuid)
        true_names.append(f'{uuid}/true')
        scaf_names.append(f'{uuid}/scaf')
        pock_names.append(f'{uuid}/pock')
        os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

    generated, starting_point = check_if_generated(output_dir, uuids, args.n_samples)
    if generated:
        print(f'Already generated batch={batch_idx}, max_uuid={max(uuids)}')
        continue
    if starting_point > 0:
        print(f'Generating {args.n_samples - starting_point} for batch={batch_idx}')

    h, x, node_mask, scaf_mask = data['one_hot'], data['positions'], data['atom_mask'], data['scaffold_mask']

    node_mask = data['atom_mask'] - data['pocket_mask']
    scaf_mask = data['scaffold_only_mask']
    pock_mask = data['pocket_mask']
    save_xyz_file_fa(output_dir, h, x, pock_mask, pock_names)
    out_xyz_pock = f'{output_dir}/{pock_names[0]}_.xyz'
    out_pdb_pock = f'{output_dir}/{pock_names[0]}_.pdb'
    subprocess.run(f'obabel {out_xyz_pock} -O {out_pdb_pock} 2> /dev/null', shell=True)

    # Save ground-truth molecules as .xyz and convert to .sdf
    save_xyz_file_fa(output_dir, h, x, node_mask, true_names)
    out_xyz_true = f'{output_dir}/{true_names[0]}_.xyz'
    out_sdf_true = f'{output_dir}/{true_names[0]}_.sdf'
    subprocess.run(f'obabel {out_xyz_true} -O {out_sdf_true} 2> /dev/null', shell=True)

    # Save scaffold as .xyz and convert to .sdf
    save_xyz_file_fa(output_dir, h, x, scaf_mask, scaf_names)
    out_xyz_scaf = f'{output_dir}/{scaf_names[0]}_.xyz'
    out_sdf_scaf = f'{output_dir}/{scaf_names[0]}_.sdf'
    subprocess.run(f'obabel {out_xyz_scaf} -O {out_sdf_scaf} 2> /dev/null', shell=True)

    # Sampling and saving generated molecules
    for i in tqdm(range(starting_point, args.n_samples), desc=str(batch_idx)):
        chain, node_mask, mean = model.sample_chain(data, sample_fn=sample_fn, keep_frames=1)#开始采样
        x = chain[-1][:, :, :model.n_dims]
        h = chain[-1][:, :, model.n_dims:]

        x += mean
        x_rgroup_tmp = x * data['rgroup_mask_batch_new']
        x_scaf_ori_tmp = data['positions'] * data['scaffold_mask']
        cnt = 0 
        for k in range(data['batch_new_len_tensor'].shape[0]):
            for j in range(data['batch_new_len_tensor'][k]):
                x_scaf_ori_tmp[k] += x_rgroup_tmp[cnt]
                cnt += 1

        h_rgroup_tmp = h * data['rgroup_mask_batch_new']
        h_scaf_ori_tmp = data['one_hot'] * data['scaffold_mask']
        cnt = 0 
        for k in range(data['batch_new_len_tensor'].shape[0]):
            for j in range(data['batch_new_len_tensor'][k]):
                h_scaf_ori_tmp[k] += h_rgroup_tmp[cnt]
                cnt += 1

        x = x_scaf_ori_tmp
        h = h_scaf_ori_tmp

        node_mask = data['atom_mask'] - data['pocket_mask']

        pred_names = [f'{uuid}/{i}' for uuid in uuids]

        save_xyz_file_fa(output_dir, h, x, node_mask, pred_names)
        for j in range(len(pred_names)):
            out_xyz = f'{output_dir}/{pred_names[j]}_.xyz'
            out_sdf = f'{output_dir}/{pred_names[j]}_.sdf'
            subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)
time_end = time.time()
print('sample time:', time_end - time_start, 's')
