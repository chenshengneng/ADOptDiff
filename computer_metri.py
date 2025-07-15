import os
import glob
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
import sascorer  # 本地需有 sascorer.py
from src.utils import disable_rdkit_logging

disable_rdkit_logging()
np.random.seed(0)

def evaluate_sdf_file(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    data = []

    for mol in suppl:
        if mol is None:
            continue
        try:
            true_mol = mol.GetProp("true_molecule")
            pred_mol = mol.GetProp("pred_molecule")
        except:
            continue
        data.append({'true_molecule': true_mol, 'pred_molecule': pred_mol})

    df = pd.DataFrame(data)

    # ------------------ Validity ------------------
    def is_valid(smi):
        return Chem.MolFromSmiles(smi) is not None
    df['valid'] = df['pred_molecule'].apply(is_valid)
    validity = df['valid'].mean() * 100

    # ------------------ Uniqueness ------------------
    df_valid = df[df['valid']]
    uniqueness = df_valid['pred_molecule'].nunique() / len(df_valid) * 100 if len(df_valid) > 0 else 0

    # ------------------ Similarity ------------------
    similarities = []
    for _, row in df_valid.iterrows():
        mol1 = Chem.MolFromSmiles(row['pred_molecule'])
        mol2 = Chem.MolFromSmiles(row['true_molecule'])
        if mol1 is None or mol2 is None:
            continue
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        sim = DataStructs.FingerprintSimilarity(fp1, fp2)
        similarities.append(sim)
    similarity = np.mean(similarities) if similarities else 0

    # ------------------ Recovery ------------------
    def norm_smi(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ''
        Chem.RemoveStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)
    df['recovered'] = df['valid'] & (df['pred_molecule'].apply(norm_smi) == df['true_molecule'].apply(norm_smi))
    recovery = df['recovered'].mean() * 100

    # ------------------ SAS Score ------------------
    sas_scores = []
    for smi in df_valid['pred_molecule']:
        mol = Chem.MolFromSmiles(smi)
        try:
            sas = sascorer.calculateScore(mol)
            sas_scores.append(sas)
        except:
            continue
    sas_avg = np.mean(sas_scores) if sas_scores else 0

    summary = {
        'file': os.path.basename(sdf_path),
        'validity': round(validity, 3),
        'uniqueness': round(uniqueness, 3),
        'similarity': round(similarity, 3),
        'recovery': round(recovery, 3),
        'sas_avg': round(sas_avg, 3)
    }

    return summary, df


def batch_evaluate_sdf_folder(folder):
    sdf_files = glob.glob(os.path.join(folder, "*.sdf"))
    all_summaries = []

    for sdf_file in sdf_files:
        print(f"🧪 Processing {os.path.basename(sdf_file)}")
        try:
            summary, df = evaluate_sdf_file(sdf_file)
            all_summaries.append(summary)

            base = os.path.splitext(os.path.basename(sdf_file))[0]
            df.to_csv(os.path.join(folder, f"{base}_detailed.csv"), index=False)
            pd.DataFrame([summary]).to_csv(os.path.join(folder, f"{base}_summary.csv"), index=False)
        except Exception as e:
            print(f"❌ Error processing {sdf_file}: {e}")
            continue

    pd.DataFrame(all_summaries).to_csv(os.path.join(folder, "overall_summary.csv"), index=False)
    print("✅ All done. Summary saved to overall_summary.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing .sdf files")
    args = parser.parse_args()
    batch_evaluate_sdf_folder(args.folder)
