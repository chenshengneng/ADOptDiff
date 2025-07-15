import os
import numpy as np
import pandas as pd
import pickle
import torch
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 假设 const 模块中定义了以下常量，您需要根据项目实际设置
class const:
    ATOM2IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3}  # 原子种类字典示例
    CHARGES = {'H': 1, 'C': 0, 'N': -1, 'O': -2}  # 电荷字典示例
    TORCH_FLOAT = torch.float32

# 定义读取 SDF 文件的生成器
def read_sdf(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
    for molecule in supplier:
        if molecule is not None:
            yield molecule

# 获取原子的 One-Hot 编码
def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

# 解析分子并提取位置信息、One-Hot 编码和电荷
def parse_molecule(mol):
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)

# 数据集类定义
class MultiRDataset_anchor:
    def __init__(self, data_path, prefix, device):
        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = MultiRDataset_anchor.preprocess(data_path, prefix, pocket_mode, device)
            torch.save(self.data, dataset_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
        rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(scaffold_path), read_sdf(rgroups_path), pockets),
            total=len(table)
        )
        for (_, row), scaffold, rgroup, pocket_data in generator:
            if not isinstance(scaffold, Chem.rdchem.Mol) or not isinstance(rgroup, Chem.rdchem.Mol):
                continue

            uuid = row['uuid']
            name = row['molecule']
            anchor_id_list = str(row['anchor']).split('|') if isinstance(row['anchor'], str) else [str(row['anchor'])]

            scaf_pos, scaf_one_hot, scaf_charges = parse_molecule(scaffold)
            fake_pos_list = [list(scaf_pos[int(anchor_id)]) for anchor_id in anchor_id_list]

            try:
                rgroup_pos, rgroup_one_hot, rgroup_charges = parse_molecule(rgroup)
            except Exception as e:
                print(f"Error parsing rgroup: {e}")
                continue

            rgroup_size_str = '|'.join(['10' for _ in anchor_id_list])

            pocket_pos, pocket_one_hot, pocket_charges = [], [], []
            for i in range(len(pocket_data[f'{pocket_mode}_types'])):
                atom_type = pocket_data[f'{pocket_mode}_types'][i]
                pos = pocket_data[f'{pocket_mode}_coord'][i]
                if atom_type == 'H':
                    continue
                pocket_pos.append(pos)
                pocket_one_hot.append(get_one_hot(atom_type, const.ATOM2IDX))
                pocket_charges.append(const.CHARGES[atom_type])

            positions = np.concatenate([scaf_pos, pocket_pos, rgroup_pos], axis=0)
            one_hot = np.concatenate([scaf_one_hot, pocket_one_hot, rgroup_one_hot], axis=0)
            charges = np.concatenate([scaf_charges, pocket_charges, rgroup_charges], axis=0)
            anchors = np.zeros_like(charges)

            anchor_values = [row['anchor']] if isinstance(row['anchor'], int) else map(int, str(row['anchor']).split('|'))
            for anchor_idx in anchor_values:
                anchors[anchor_idx] = 1

            scaf_only_mask = np.concatenate([np.ones_like(scaf_charges), np.zeros_like(pocket_charges), np.zeros_like(rgroup_charges)])
            pocket_mask = np.concatenate([np.zeros_like(scaf_charges), np.ones_like(pocket_charges), np.zeros_like(rgroup_charges)])
            rgroup_mask = np.concatenate([np.zeros_like(scaf_charges), np.zeros_like(pocket_charges), np.ones_like(rgroup_charges)])
            scaf_mask = np.concatenate([np.ones_like(scaf_charges), np.ones_like(pocket_charges), np.zeros_like(rgroup_charges)])

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_only_mask': torch.tensor(scaf_only_mask, dtype=const.TORCH_FLOAT, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_mask': torch.tensor(scaf_mask, dtype=const.TORCH_FLOAT, device=device),
                'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
                'rgroup_size': rgroup_size_str,
                'anchors_str': row['anchor'],
            })

        return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process and save dataset.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--prefix', type=str, required=True, help="Prefix for the dataset.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to load data on.")
    args = parser.parse_args()

    dataset = MultiRDataset_anchor(data_path=args.data_path, prefix=args.prefix, device=args.device)
    print(f"Processed dataset with {len(dataset)} molecules.")
