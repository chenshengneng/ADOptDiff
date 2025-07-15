from rdkit import Chem
from collections import defaultdict
import argparse
import os
import pandas as pd
import glob

def get_ring_size_distribution(sdf_path, out_path):
    # === Step 1: 收集所有 SDF 文件路径 ===
    premol_sdf_dirs = []
    # for file_name in os.listdir(sdf_path):    # 遍历每个样本文件夹,pocket2mols
    #     parent = os.path.join(sdf_path, file_name)
    #     if os.path.isdir(parent):
    #         for sub in os.listdir(parent):
    #             full_path = os.path.join(parent, sub)
    #             if os.path.isdir(full_path) and sub == 'SDF':
    #                 for gen_sdf_file in sorted(glob.glob(os.path.join(full_path, '[0-9]*.sdf'))):
    #                     premol_sdf_dirs.append(gen_sdf_file)
    for file_name in os.listdir(sdf_path):#FLAg的验证
        full_path = os.path.join(sdf_path, file_name)
        if os.path.isdir(full_path):
            for gen_sdf_file in sorted(glob.glob(os.path.join(full_path, '[0-9]*.sdf'))):
                premol_sdf_dirs.append(gen_sdf_file)
    # print(premol_sdf_dirs)
    # exit()            


    total_mols = 0
    ring_size_counter = defaultdict(int)

    # === Step 2: 遍历 SDF 文件，统计环大小 ===
    for sdf_file in premol_sdf_dirs:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            total_mols += 1
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            unique_sizes = set([len(r) for r in atom_rings])
            for size in unique_sizes:
                ring_size_counter[size] += 1

    # === Step 3: 无有效分子时处理 ===
    if total_mols == 0:
        print("[!] No valid molecules found in the dataset.")
        return

    # === Step 4: 构建 DataFrame，保存为 CSV ===
    result = []
    for size in sorted(ring_size_counter):
        ratio = ring_size_counter[size] / total_mols
        result.append({
            "Ring Size": size,
            "Count": ring_size_counter[size],
            "Frequency": ratio
        })

    df = pd.DataFrame(result)

    # 如果输出是目录，自动加文件名
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "ring_distribution.csv")

    df.to_csv(out_path, index=False)
    print(f"[✓] Saved ring size distribution to: {out_path}")

    # 控制台打印结果
    print(f"Total molecules: {total_mols}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ring size distribution from Pocket2Mol output")
    parser.add_argument("--sdf", type=str, required=True, help="Path to Pocket2Mol outputs dir")
    parser.add_argument("--out", type=str, default="ring_distribution.csv", help="Output CSV path or directory")
    args = parser.parse_args()

    get_ring_size_distribution(args.sdf, args.out)
