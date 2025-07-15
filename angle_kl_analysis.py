import os
import glob
import argparse
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.stats import entropy
import numpy as np
import pandas as pd

# === 结构模式定义（键角 + 二面角）===
ANGLE_SMARTS = {
    'CCC': '[C:1]-[C:2]-[C:3]',
    'CCO': '[C:1]-[C:2]-[O:3]',
    'CNC': '[C:1]-[N:2]-[C:3]',
    'OPO': '[O:1]-[P:2]-[O:3]',
    'NCC': '[N:1]-[C:2]-[C:3]',
    'CC=O': '[C:1]-[C:2]=[O:3]',
    'COC': '[C:1]-[O:2]-[C:3]',

    'CCCC': '[C:1]-[C:2]-[C:3]-[C:4]',
    'cccc': '[c:1]-[c:2]-[c:3]-[c:4]',
    'OCCO': '[O:1]-[C:2]-[C:3]-[O:4]',
    'OCCCO': '[O:1]-[C:2]-[C:3]-[C:4]',
    'Cccc': '[C:1]-[c:2]-[c:3]-[c:4]',
    'CC=CC': '[C:1]-[C:2]=[C:3]-[C:4]',
}

# === 提取角度值 ===
def extract_angles(mol, smarts, angle_type):
    patt = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(patt)
    values = []
    conf = mol.GetConformer()
    for match in matches:
        try:
            if angle_type == 'angle':
                val = rdMolTransforms.GetAngleDeg(conf, *match)
            else:
                val = rdMolTransforms.GetDihedralDeg(conf, *match)
            values.append(val)
        except:
            continue
    return values

# === 收集角度分布 ===
def collect_distribution(sdf_path, debug=False):
    data = defaultdict(list)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    valid_mol_count = 0

    for mol in suppl:
        if mol is None:
            if debug:
                print(f"[WARN] Failed to load mol in {sdf_path}")
            continue
        if not mol.GetNumConformers():
            if debug:
                print(f"[WARN] Mol in {sdf_path} has no conformer")
            continue

        valid_mol_count += 1
        for label, smarts in ANGLE_SMARTS.items():
            angle_type = 'angle' if len(label) == 3 else 'torsion'
            values = extract_angles(mol, smarts, angle_type)
            data[label].extend(values)

    if debug:
        print(f"[INFO] {sdf_path}: {valid_mol_count} valid mols")
        for k in data:
            print(f"    - {k}: {len(data[k])} values")
    return data

# === KL 散度计算 ===
def compute_kl(p_vals, q_vals, bins=180, range_=(0, 180)):
    if len(p_vals) == 0 or len(q_vals) == 0:
        return np.nan
    if range_[1] - range_[0] == 360:
        bins = 360
    p_hist, _ = np.histogram(p_vals, bins=bins, range=range_, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, range=range_, density=True)
    p_hist += 1e-8
    q_hist += 1e-8
    return entropy(p_hist, q_hist)

# === 主评估函数 ===
def evaluate_kl(test_sdf_csv, gen_root_path, root_dir='/public/home/chensn/diffdec/DiffDec-master/try_model_data/', debug=False):
    df = pd.read_csv(test_sdf_csv)
    index = list(df[['ligand_path']].itertuples(index=False, name=None))

    # --- 1. 聚合测试集角度 ---
    test_data = defaultdict(list)
    for test_sdf_file in index:
        file_sdf = os.path.join(root_dir, test_sdf_file[0])
        dist = collect_distribution(file_sdf, debug=debug)
        for key, values in dist.items():
            test_data[key].extend(values)

    # --- 2. 聚合生成分子角度 ---
    gen_data = defaultdict(list)
    premol_sdf_dirs = []
    # for file_name in os.listdir(gen_root_path):    #pocket2mol的验证
    #     parent = os.path.join(gen_root_path, file_name)
    #     if os.path.isdir(parent):
    #         for sub in os.listdir(parent):
    #             full_path = os.path.join(parent, sub)
    #             if os.path.isdir(full_path) and sub == 'SDF':
    #                 premol_sdf_dirs.append(full_path)
    # print(premol_sdf_dirs)
    # exit()
    for file_name in os.listdir(gen_root_path):#FLAg的验证
        full_path = os.path.join(gen_root_path, file_name)
        if os.path.isdir(full_path):
            premol_sdf_dirs.append(full_path)
    # # print(premol_sdf_dirs)
    # # exit()

    for sub in sorted(premol_sdf_dirs):
        for gen_sdf_file in sorted(glob.glob(os.path.join(sub, '[0-9]*.sdf'))):
            dist = collect_distribution(gen_sdf_file, debug=debug)
            for key, values in dist.items():
                gen_data[key].extend(values)

    # --- 3. KL 计算 ---
    results = []
    for key in ANGLE_SMARTS.keys():
        if key not in test_data or key not in gen_data:
            if debug:
                print(f"[SKIP] {key} not in both distributions.")
            continue
        angle_type = 'angle' if len(key) == 3 else 'torsion'
        kl = compute_kl(test_data[key], gen_data[key],
                        range_=(0, 180) if angle_type == 'angle' else (-180, 180))
        results.append((key, kl))

    df_kl = pd.DataFrame(results, columns=["Angle_Type", "KL_Divergence"])
    return df_kl

# === 命令行入口 ===
def main():
    parser = argparse.ArgumentParser(description="Calculate KL divergence between test and generated molecules")
    parser.add_argument('--test', type=str, required=True, help='CSV path containing ligand_path column')
    parser.add_argument('--gen', type=str, required=True, help='Root directory containing generated SDF folders')
    parser.add_argument('--out', type=str, default='kl_results.csv', help='Output CSV file or directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    df = evaluate_kl(args.test, args.gen, debug=args.debug)

    out_path = args.out
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "kl_results.csv")

    df.to_csv(out_path, index=False)
    print(f"[✓] KL divergence results saved to: {out_path}")
    print(df if not df.empty else "[!] KL result is empty — check if angle types were extracted.")

if __name__ == '__main__':
    main()
