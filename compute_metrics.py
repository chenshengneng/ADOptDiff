import os
import glob
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
import sascorer
from src.utils import disable_rdkit_logging

disable_rdkit_logging()
np.random.seed(0)

MAX_MOLS = 100  # 固定每个文件夹参与评估的预测分子数量

def extract_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return ''

def evaluate_single_folder(folder):
    true_sdf_path = os.path.join(folder, 'mol.sdf')
    true_sdf = Chem.SDMolSupplier(true_sdf_path)
    true_mol = true_sdf[0] if len(true_sdf) > 0 else None
    if true_mol is None:
        raise ValueError(f"❌ 无法读取 true_molecule in {folder}")
    true_smi = extract_smiles(true_mol)

    pred_smi_list = []
    for file in sorted(glob.glob(os.path.join(folder, '[0-9]*.sdf'))):
        mols = Chem.SDMolSupplier(file)
        mol = mols[0] if len(mols) > 0 else None
        if mol is not None:
            pred_smi = extract_smiles(mol)
            if pred_smi:
                pred_smi_list.append(pred_smi)
            if len(pred_smi_list) >= MAX_MOLS:
                break

    pred_smi_list = pred_smi_list[:MAX_MOLS]  # 限制最多 MAX_MOLS 个分子

    valid_flags = [Chem.MolFromSmiles(smi) is not None for smi in pred_smi_list]
    valid_smiles = [smi for smi, flag in zip(pred_smi_list, valid_flags) if flag]

    validity = len(valid_smiles) / MAX_MOLS * 100
    uniqueness = len(set(valid_smiles)) / len(valid_smiles) * 100 if valid_smiles else 0

    similarities = []
    true_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(true_smi))
    for smi in valid_smiles:
        pred_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
        sim = DataStructs.FingerprintSimilarity(pred_fp, true_fp)
        similarities.append(sim)
    # 若不足 MAX_MOLS，用最差相似度 0 填充
    if len(similarities) < MAX_MOLS:
        similarities.extend([0.0] * (MAX_MOLS - len(similarities)))
    similarity = np.mean(similarities)

    def norm_smi(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ''
        Chem.RemoveStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)

    norm_true = norm_smi(true_smi)
    rec_flags = [norm_smi(s) == norm_true for s in valid_smiles]
    # 若不足 MAX_MOLS，用错误（False）填充
    if len(rec_flags) < MAX_MOLS:
        rec_flags.extend([False] * (MAX_MOLS - len(rec_flags)))
    recovery = sum(rec_flags) / MAX_MOLS * 100

    sas_scores = []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        try:
            sas = sascorer.calculateScore(mol)
            sas_scores.append(sas)
        except:
            continue

    # 若有效分子数量不足 MAX_MOLS，用最高分 10 填补
    if len(sas_scores) < MAX_MOLS:
        sas_scores.extend([10.0] * (MAX_MOLS - len(sas_scores)))

    sas_avg = np.mean(sas_scores)

    return {
        'folder': os.path.basename(folder),
        'validity': round(validity, 3),
        'uniqueness': round(uniqueness, 3),
        'similarity': round(similarity, 3),
        'recovery': round(recovery, 3),
        'sas_avg': round(sas_avg, 3)
    }

def batch_evaluate_all_subfolders(root_dir):
    sdf_dirs = []
    for file_name in os.listdir(root_dir):
        parent = os.path.join(root_dir, file_name)
        sdf_dirs.append(parent)
        # if os.path.isdir(parent):
        #     for sub in os.listdir(parent):
        #         full_path = os.path.join(parent, sub)
        #         if os.path.isdir(full_path) and sub == 'SDF':
        #             sdf_dirs.append(full_path)

    all_summaries = []

    for sub in sorted(sdf_dirs):
        try:
            print(f"🧪 Evaluating {sub} ...")
            result = evaluate_single_folder(sub)
            all_summaries.append(result)
        except Exception as e:
            print(f"❌ Error in {sub}: {e}")

    if not all_summaries:
        print("⚠️ No valid results to summarize.")
        return

    df = pd.DataFrame(all_summaries)
    df.to_csv(os.path.join(root_dir, "overall_summary.csv"), index=False)

    avg_row = {
        'folder': 'AVERAGE',
        'validity': round(df['validity'].mean(), 3),
        'uniqueness': round(df['uniqueness'].mean(), 3),
        'similarity': round(df['similarity'].mean(), 3),
        'recovery': round(df['recovery'].mean(), 3),
        'sas_avg': round(df['sas_avg'].mean(), 3)
    }
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df_with_avg.to_csv(os.path.join(root_dir, "overall_summary_with_average.csv"), index=False)
    print("✅ 平均值写入完成，保存在 overall_summary_with_average.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="根文件夹，每个子文件夹含 true + pred sdf")
    args = parser.parse_args()

    batch_evaluate_all_subfolders(args.root)
