import os
import subprocess
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import SDWriter
import random
import string
import argparse
import csv
import json
import shlex
import shutil
import sascorer
import glob

print(sascorer.__file__)

def compute_sas_from_smiles(smiles: str):
    """计算 SMILES 的 SAS 评分"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        score = sascorer.calculateScore(mol)
        return score  # SAS 评分通常在 1~10 之间，越低越容易合成
    except Exception as e:
        print(f"Error calculating SAS for SMILES {smiles}: {e}")
        return None

class BaseDockingTask(object):
    def __init__(self, pdb_block, ligand_rdmol):
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()

def get_random_id(length=10):
    """Generate a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_data(cls, ligand_mol, protein_path, ligand_path):
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        struct = PDBParser().get_structure('', protein_path)
        return cls(pdb_block, ligand_mol, ligand_path, struct)

    def __init__(self, pdb_block, ligand_rdmol, ligand_path, struct, conda_env='myDiffDec', tmp_dir='./tmp_true', center=None):
        super().__init__(pdb_block, ligand_rdmol)

        residue_ids = []
        atom_coords = []

        for residue in struct.get_residues():
            resid = residue.get_id()[1]
            for atom in residue.get_atoms():
                atom_coords.append(atom.get_coord())
                residue_ids.append(resid)

        residue_ids = np.array(residue_ids)
        atom_coords = np.array(atom_coords)
        center_pro = (atom_coords.max(0) + atom_coords.min(0)) / 2

        if ligand_rdmol.GetNumConformers() == 0:
            raise ValueError("The ligand molecule must have a 3D conformer to retain its original pose.")

        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        sdf_writer = SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()

        self.ligand_rdmol = ligand_rdmol
        pos = ligand_rdmol.GetConformer(0).GetPositions()
        self.center = (pos.max(0) + pos.min(0)) / 2 if center is None else center
        self.center = center_pro

        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_path = None

    def run(self, exhaustiveness=100, score_only=False):
        score_flag = "--score_only" if score_only else f"""
            --center_x {self.center[0]:.4f} \
            --center_y {self.center[1]:.4f} \
            --center_z {self.center[2]:.4f} \
            --size_x 60 --size_y 60 --size_z 60 \
            --exhaustiveness {exhaustiveness}
        """

        commands = f"""
eval \"$(conda shell.bash hook)\" 
conda activate {self.conda_env} 
cd {self.tmp_dir} 
# Prepare receptor (PDB->PDBQT)
/public/home/chensn/DL/Diffdec/autodocktools-prepare-py3k-master/prepare_receptor4.py -r {self.receptor_id}.pdb -o {self.receptor_id}.pdbqt
# Prepare ligand
obabel {self.ligand_id}.sdf -O{self.ligand_id}.pdbqt --partialcharge  
qvina \
    --receptor {self.receptor_id}.pdbqt \
    --ligand {self.ligand_id}.pdbqt \
    {score_flag} \
    --cpu 32 \
    --seed 1
obabel {self.ligand_id}_out.pdbqt -O{self.ligand_id}_out.sdf -h
        """

        self.docked_sdf_path = os.path.join(self.tmp_dir, f'{self.ligand_id}_out.sdf')
        self.proc = subprocess.Popen(
            commands,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        #stdout, stderr = self.proc.communicate(commands.encode('utf-8'))
        stdout, stderr = self.proc.communicate()
        self.output = stdout.decode()
        self.errors = stderr.decode()
        #if "Parse error" in self.errors:
            #print(f"Parse error detected in {self.ligand_id}.pdbqt: {self.errors}")
            #self.output = None  # 清空 output 表示任务失败
            #return
        if "Parse error" in self.errors:
            print(f"Parse error detected: {self.errors}")
            self.output = None  # 清空 output 表示任务失败
            return
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        #receptor_pdbqt_path = os.path.join(self.tmp_dir, f"{self.receptor_id}.pdbqt")
        #ligand_pdbqt_path = os.path.join(self.tmp_dir, f"{self.ligand_id}.pdbqt")
        #print(f"Receptor PDBQT file: {receptor_pdbqt_path}")
        #print(f"Ligand PDBQT file: {ligand_pdbqt_path}")

        #self.proc.stdin.write(commands.encode('utf-8'))
        #self.proc.stdin.close()
        #if os.path.exists(self.docked_sdf_path):
           # print(f"Docking results saved at: {self.docked_sdf_path}")
            #self._save_docked_molecule()
       # else:
            #print("Docking failed. Output file not found.")

    #def _save_docked_molecule(self):
        #"""Save and print the docked molecule."""
        #docked_mol = Chem.SDMolSupplier(self.docked_sdf_path)[0]
        #if docked_mol:
           # output_sdf_path = os.path.join(self.tmp_dir, f'docked_{self.ligand_id}.sdf')
           # with SDWriter(output_sdf_path) as writer:
                #writer.write(docked_mol)
            #print(f"Docked molecule saved to: {output_sdf_path}")
       # else:
           # print("Failed to load docked molecule.")
        

    def run_sync(self, exhaustiveness=100, score_only=False):
        self.run(exhaustiveness=exhaustiveness, score_only=score_only)
        while self.get_results() is None:
            pass
        return self.get_results()

    def get_results(self):
        if self.output is None:
            return None

        results = []
        for line in self.output.splitlines():
            if line.startswith("Affinity:"):
                try:
                    affinity = float(line.split()[1])
                    results.append({'affinity': affinity})
                except ValueError as e:
                    print(f"Error parsing affinity: {e}")
        return results if results else None

def clogD(logP, pKa, pH=7.4):
    """
     logD :
    logD = logP - log10(1 + 10^(pH - pKa))
    """
    if pKa is None:
        return logP                                  # 如果没有 pKa 数据，logD 直接等于 logP
    return logP - log10(1 + 10 ** (pH - pKa))

def mw_score(mw):
    """ MW """
    if mw is None:
        return 0
    if mw <= 360:
        return 1
    elif 360 < mw <= 500:
        return -0.005 * mw + 2.5
    else:
        return 0

def logp_score(logp):
    """ logP """
    if logp is None:
        return 0
    if logp <= 3:
        return 1
    elif 3 < logp <= 5:
        return -0.5 * logp + 2.5
    else:
        return 0

def logd_score(logd):
    """ logD """
    if logd is None:
        return 0
    if logd <= 2:
        return 1
    elif 2 < logd <= 4:
        return -0.5 * logd + 2
    else:
        return 0

def pka_score(pka):
    """ pKa """
    if pka is None:
        return 0
    if pka <= 8:
        return 1
    elif 8 < pka <= 10:
        return -0.5 * pka + 5
    else:
        return 0

def tpsa_score(tpsa):
    """ TPSA """
    if tpsa is None:
        return 0
    if 40 <= tpsa <= 90:
        return 1
    elif 90 < tpsa <= 120:
        return -0.0333 * tpsa + 4
    elif 20 <= tpsa < 40:
        return 0.05 * tpsa - 1
    else:
        return 0

def hbd_score(hbd):
    """ HBD """
    if hbd is None:
        return 0
    if hbd == 0:
        return 1
    elif hbd == 1:
        return 0.75
    elif hbd == 2:
        return 0.5
    elif hbd == 3:
        return 0.25
    else:
        return 0

def calculate_cns_mpo(properties):
    """
    计算 CNS-MPO 评分，输入 `properties` 为包含分子性质的字典
    """
    # 计算 logD
    logP = properties.get("logP", None)
    pKa = properties.get("pKa", None)
    logD = clogD(logP, pKa) if logP is not None else None

    # 计算评分
    scores = {
        "MW": mw_score(properties.get("MW", None)),
        "LogP": logp_score(logP),
        "LogD": logd_score(logD),
        "pKa": pka_score(pKa),
        "TPSA": tpsa_score(properties.get("TPSA", None)),
        "HBD": hbd_score(properties.get("HBD", None)),
    }
    
    # 计算 CNS-MPO 总分
    cns_mpo_score = sum(scores.values())

    # 返回详细信息
    return {
        "CNS-MPO Score": round(cns_mpo_score, 2),
        "Scores": scores
    }



def run_cns_mpo_calculation(smiles, cns_mpo_env='cns_mpo'):
    """
    在 `cns_mpo` Conda 环境中运行 `test.py` 计算分子性质
    """
    command = f"""
    eval "$(conda shell.bash hook)"
    conda activate {cns_mpo_env}
    python /public/home/chensn/mol_property/test.py {shlex.quote(smiles)}
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:  # 只有进程失败时才打印错误
        print(f"Error running test.py: {stderr.decode()}")
        return None
    
    try:
        output = stdout.decode().strip()
        if not output:  # 确保 stdout 不是空的
            print("Error: No output from test.py")
            return None
        return json.loads(output)
    except json.JSONDecodeError:
        print(f"Error parsing JSON output from test.py: {output}")
        return None
def dock_and_save_cns_mpo_greater_than_4(sdf_dir, pdb_file, output_dir, cns_mpo_env='cns_mpo', score_only=False, cns_sas_output_dir='/public/home/chensn/DL/DiffDec-master/CNS', docking_results_csv_dir='/public/home/chensn/DL/DiffDec-master/docking_results_csv'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cns_sas_output_dir, exist_ok=True)
    os.makedirs(docking_results_csv_dir, exist_ok=True)

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]

    # For docking results CSV
    docking_results_csv_path = os.path.join(docking_results_csv_dir, "vina_scores.csv")
    with open(docking_results_csv_path, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["SDF File", "Affinity (kcal/mol)", "CNS_MPO", "SAS Score","Smiles"])

        for sdf_file in sdf_files:
            sdf_path = os.path.join(sdf_dir, sdf_file)
            print(f"Processing SDF file: {sdf_file}")

            ligand_rdmol = Chem.SDMolSupplier(sdf_path)[0]
            if ligand_rdmol is None:
                print(f"Skipping {sdf_file}, invalid molecule.")
                continue

            try:
                task = QVinaDockingTask.from_data(
                    ligand_mol=ligand_rdmol,
                    protein_path=pdb_file,
                    ligand_path=sdf_path,
                )

                task.run(score_only=score_only)
                results = task.get_results()

                # Get affinity and CNS-MPO score
                affinity = results[0]['affinity'] if results else None
                # mol = next(ligand_rdmol)
                # Chem.SanitizeMol(mol)
                # smiles = Chem.MolToSmiles(mol, canonical=True)
                #print(f"Processed SMILES: {smiles}")
                Chem.SanitizeMol(ligand_rdmol)
                smiles = Chem.MolToSmiles(ligand_rdmol, canonical=True)
                sas_smiles = smiles.replace("[", "").replace("]", "")
                # print(f"Processed SMILES: {smiles}")
                # print(f"Processed SMILES: {sas_smiles}")
                # exit()
                properties = run_cns_mpo_calculation(smiles, cns_mpo_env)
                cns_mpo = calculate_cns_mpo(properties) if properties else None
                #sas_mol = Chem.MolFromSmiles(smiles)
                sas_score = compute_sas_from_smiles(sas_smiles)

                if cns_mpo and sas_score is not None:
                    #if cns_mpo['CNS-MPO Score'] >= 4 and sas_score < 4:
                    if sas_score < 4:
                        shutil.copy(sdf_path, os.path.join(cns_sas_output_dir, sdf_file))
                        #shutil.copy(sdf_path, os.path.join(sas_output_dir, sdf_file))
                        print(f"Copy {sdf_file} to both CNS and SAS directories.")

                # if cns_mpo and cns_mpo['CNS-MPO Score'] > 4:
                #     # Move the SDF file to the CNS directory if CNS-MPO > 4
                #     shutil.copy(sdf_path, os.path.join(cns_output_dir, sdf_file))
                #     print(f"Copy {sdf_file} to CNS directory.")

                #     # Write the docking results to CSV
                csvwriter.writerow([sdf_file, affinity, cns_mpo['CNS-MPO Score'], sas_score, sas_smiles])
                
            except Exception as e:
                print(f"Error docking {sdf_file}: {e}")

def get_affinity(mol, protein_path):
    try:
        task = QVinaDockingTask.from_data(
            ligand_mol=mol,
            protein_path=protein_path,
            ligand_path=None
        )
        task.run(score_only=True)
        results = task.get_results()
        if results and 'affinity' in results[0]:
            return results[0]['affinity']
    except Exception as e:
        print(f"⚠️  Docking error: {e}")
    return None

def compute_vina_stats(folder_path):
    protein_path = os.path.join(folder_path, 'protein.pdb')
    mol_path = os.path.join(folder_path, 'mol.sdf')

    if not os.path.exists(protein_path) or not os.path.exists(mol_path):
        print(f"❌ Missing mol.sdf or protein.pdb in {folder_path}")
        return None

    true_mol = Chem.SDMolSupplier(mol_path)[0]
    if true_mol is None:
        print(f"❌ Invalid mol.sdf in {folder_path}")
        return None

    mol_affinity = get_affinity(true_mol, protein_path)
    if mol_affinity is None:
        print(f"❌ Failed to get affinity for mol.sdf in {folder_path}")
        return None

    pre_affinities = []
    pre_mol_files = sorted(glob.glob(os.path.join(folder_path, '[0-9]*.sdf')))

    for pre_mol_path in pre_mol_files:
        pre_mols = [m for m in Chem.SDMolSupplier(pre_mol_path) if m is not None]
        for mol in pre_mols:
            if len(pre_affinities) >= 100:
                break
            affinity = get_affinity(mol, protein_path)
            if affinity is not None:
                pre_affinities.append(affinity)
        if len(pre_affinities) >= 100:
            break

    # 按 100 为分母计算 better_ratio（即使 pre_affinities 不足 100 个）
    better_count = sum([1 for aff in pre_affinities if aff < mol_affinity])
    better_ratio = better_count / 100.0

    avg_pre_aff = sum(pre_affinities) / len(pre_affinities) if pre_affinities else None

    return {
        'mol_affinity': mol_affinity,
        'pre_avg_affinity': round(avg_pre_aff, 3) if avg_pre_aff is not None else None,
        'better_than_mol_ratio': round(better_ratio, 3),
        'pre_affinities': pre_affinities,
        'better_count': better_count
    }

def evaluate_all(root_dir, output_csv='vina_stats.csv'):
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    output_path = os.path.join(root_dir, output_csv)

    all_pre_affinities = []
    total_better_than_mol = 0
    folder_evaluated = 0

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['folder', 'mol_affinity', 'pre_avg_affinity', 'better_than_mol_ratio'])

        for sub in sorted(subfolders):
            #folder_path = os.path.join(root_dir, sub, 'SDF')
            folder_path = os.path.join(root_dir, sub)
            if not os.path.exists(folder_path):
                 continue

            print(f"🧪 Evaluating {sub}...")
            # stats = compute_vina_stats(folder_path)
            stats = compute_vina_stats(folder_path)

            if stats:
                folder_evaluated += 1
                writer.writerow([
                    sub,
                    round(stats['mol_affinity'], 3),
                    stats['pre_avg_affinity'] if stats['pre_avg_affinity'] is not None else 'N/A',
                    stats['better_than_mol_ratio']
                ])

                all_pre_affinities.extend(stats['pre_affinities'])
                total_better_than_mol += stats['better_count']

    print(f"\n✅ 结果保存在 {output_path}")

    if folder_evaluated > 0 and all_pre_affinities:
        # total_avg = sum(all_pre_affinities) / len(all_pre_affinities)
        total_avg = sum(all_pre_affinities) / (folder_evaluated * 100)
        global_ratio = total_better_than_mol / (folder_evaluated * 100)

        print(f"\n📊 所有 pre-mol 分子平均打分：{round(total_avg, 3)}")
        print(f"📈 高于对应 mol 的占比（按固定100个计）：{round(global_ratio * 100, 2)}%")
    else:
        print("⚠️ 无有效数据用于统计")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, help='根目录，包含多个子目录，每个子目录中有 SDF/ 结构文件')
    args = parser.parse_args()

    evaluate_all(args.root_dir)
# 为了公平的
# def get_affinity(mol, protein_path):
#     try:
#         task = QVinaDockingTask.from_data(
#             ligand_mol=mol,
#             protein_path=protein_path,
#             ligand_path=None
#         )
#         task.run(score_only=True)
#         results = task.get_results()
#         if results and 'affinity' in results[0]:
#             return results[0]['affinity']
#     except Exception as e:
#         print(f"⚠️  Docking error: {e}")
#     return None

# def compute_vina_stats(folder_path):
#     protein_path = os.path.join(folder_path, 'protein.pdb')
#     mol_path = os.path.join(folder_path, 'mol.sdf')

#     if not os.path.exists(protein_path) or not os.path.exists(mol_path):
#         print(f"❌ Missing mol.sdf or protein.pdb in {folder_path}")
#         return None

#     true_mol = Chem.SDMolSupplier(mol_path)[0]
#     if true_mol is None:
#         print(f"❌ Invalid mol.sdf in {folder_path}")
#         return None

#     mol_affinity = get_affinity(true_mol, protein_path)
#     if mol_affinity is None:
#         print(f"❌ Failed to get affinity for mol.sdf in {folder_path}")
#         return None

#     pre_affinities = []
#     pre_mol_files = sorted(glob.glob(os.path.join(folder_path, '[0-9]*.sdf')))

#     for pre_mol_path in pre_mol_files:
#         pre_mols = [m for m in Chem.SDMolSupplier(pre_mol_path) if m is not None]
#         for mol in pre_mols:
#             affinity = get_affinity(mol, protein_path)
#             if affinity is not None:
#                 pre_affinities.append(affinity)

#     if len(pre_affinities) == 0:
#         return {
#             'mol_affinity': mol_affinity,
#             'pre_avg_affinity': None,
#             'better_than_mol_ratio': None,
#             'pre_affinities': []
#         }

#     avg_pre_aff = sum(pre_affinities) / len(pre_affinities)
#     #better_ratio = sum([1 for aff in pre_affinities if aff < mol_affinity]) / len(pre_affinities)
#     better_count = sum([1 for aff in pre_affinities if aff < mol_affinity])
#     better_ratio = better_count / 100.0

#     return {
#         'mol_affinity': mol_affinity,
#         'pre_avg_affinity': round(avg_pre_aff, 3),
#         'better_than_mol_ratio': round(better_ratio, 3),
#         'pre_affinities': pre_affinities
#     }

# def evaluate_all(root_dir, output_csv='vina_stats.csv'):
#     subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
#     output_path = os.path.join(root_dir, output_csv)

#     all_pre_affinities = []
#     better_than_mol_total = 0
#     total_pre_mol_count = 0

#     with open(output_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['folder', 'mol_affinity', 'pre_avg_affinity', 'better_than_mol_ratio'])

#         for sub in sorted(subfolders):
#             folder_path = os.path.join(root_dir, sub)
#             sdf_dir = os.path.join(folder_path, "SDF")
#             print(f"🧪 Evaluating {sub}...")
#             stats = compute_vina_stats(sdf_dir)

#             if stats:
#                 writer.writerow([
#                     sub,
#                     round(stats['mol_affinity'], 3),
#                     stats['pre_avg_affinity'] if stats['pre_avg_affinity'] is not None else 'N/A',
#                     stats['better_than_mol_ratio'] if stats['better_than_mol_ratio'] is not None else 'N/A'
#                 ])

#                 if stats.get('pre_affinities') is not None:
#                     all_pre_affinities.extend(stats['pre_affinities'])
#                     mol_score = stats['mol_affinity']
#                     better_count = sum([1 for aff in stats['pre_affinities'] if aff < mol_score])
#                     better_than_mol_total += better_count
#                     total_pre_mol_count += len(stats['pre_affinities'])

#     print(f"✅ 结果保存在 {output_path}")

#     if all_pre_affinities:
#         total_avg = sum(all_pre_affinities) / len(all_pre_affinities)
#         global_ratio = better_than_mol_total / total_pre_mol_count if total_pre_mol_count > 0 else 0
#         print(f"\n📊 所有 pre-mol 分子平均打分：{round(total_avg, 3)}")
#         print(f"📈 高于对应 mol 的占比：{round(global_ratio * 100, 2)}%")

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root_dir', required=True, help='根目录，包含多个子目录，每个目录中含 mol.sdf、pre-mol.sdf、protein.pdb')
#     args = parser.parse_args()

#     evaluate_all(args.root_dir)

#原始的


# def dock_all_sdfs_to_pdb(sdf_dir, pdb_file, output_dir, conda_env='myDiffDec', output_csv="results.csv",cns_mpo_env='cns_mpo', score_only=False):
#     os.makedirs(output_dir, exist_ok=True)

#     sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]

#     with open(output_csv, mode='w', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(["SDF File", "Affinity (kcal/mol)", "CNS_MPO"])

#         for sdf_file in sdf_files:
#             sdf_path = os.path.join(sdf_dir, sdf_file)
#             print(f"Processing SDF file: {sdf_file}")

#             ligand_rdmol = Chem.SDMolSupplier(sdf_path)[0]
#             if ligand_rdmol is None:
#                 print(f"Skipping {sdf_file}, invalid molecule.")
#                 csvwriter.writerow([sdf_file, None, "Invalid molecule"])
#                 continue

#             try:
#                 task = QVinaDockingTask.from_data(
#                     ligand_mol=ligand_rdmol,
#                     protein_path=pdb_file,
#                     ligand_path=sdf_path,
#                 )

#                 #results = task.run_sync(score_only=score_only)
#                 task.run(score_only=score_only)
#                 results = task.get_results()
#                 # if results:
#                 #     best_result = results[0]
#                 #     csvwriter.writerow([sdf_file, best_result['affinity'], None])
#                 # else:
#                 #     csvwriter.writerow([sdf_file, None, "No results"])

#                 affinity = results[0]['affinity'] if results else None
#                 smiles = Chem.MolToSmiles(ligand_rdmol)
#                 properties = run_cns_mpo_calculation(smiles, cns_mpo_env)
#                 #print(properties)
#                 #exit()
#                 cns_mpo = calculate_cns_mpo(properties) if properties else None

#                 csvwriter.writerow([sdf_file, affinity, cns_mpo, None])
#             except Exception as e:
#                 # 捕获任务失败并记录错误
#                 print(f"Error docking {sdf_file}: {e}")
#                 if task.errors:
#                     csvwriter.writerow([sdf_file, None, task.errors])
#                 else:
#                     csvwriter.writerow([sdf_file, None, str(e)])

# if __name__ == "__main__":
#     sdf_directory = "/public/home/chensn/DL/DiffDec-master/samples_exper_c_2_19/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512/0"#生成的分子
#     pdb_file_path = "/public/home/chensn/DL/DiffDec-master/samples_exper_c_2_19/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512/0/pock_.pdb"#靶点
#     output_directory = "/public/home/chensn/DL/DiffDec-master/docking_results_2_19"
#     output_csv_path = os.path.join(output_directory, "results.csv")

#     dock_all_sdfs_to_pdb(
#         sdf_dir=sdf_directory,
#         pdb_file=pdb_file_path,
#         output_dir=output_directory,
#         output_csv=output_csv_path,
#         score_only=True  # Enable score_only mode
#     )