import os
import subprocess
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import SDWriter
import random
import string
import csv

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
/public/home/chensn/DL/DiffDec-master/autodocktools-prepare-py3k-master/prepare_receptor4.py -r {self.receptor_id}.pdb -o {self.receptor_id}.pdbqt
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

        #self.proc = subprocess.Popen(
            #'/bin/bash',
            #shell=False,
            #stdin=subprocess.PIPE,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE
        #)
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

def dock_all_sdfs_to_pdb(sdf_dir, pdb_file, output_dir, conda_env='myDiffDec', output_csv="results.csv", score_only=False):
    os.makedirs(output_dir, exist_ok=True)

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]

    with open(output_csv, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["SDF File", "Affinity (kcal/mol)", "Error"])

        for sdf_file in sdf_files:
            sdf_path = os.path.join(sdf_dir, sdf_file)
            print(f"Processing SDF file: {sdf_file}")

            ligand_rdmol = Chem.SDMolSupplier(sdf_path)[0]
            if ligand_rdmol is None:
                print(f"Skipping {sdf_file}, invalid molecule.")
                csvwriter.writerow([sdf_file, None, "Invalid molecule"])
                continue

            try:
                task = QVinaDockingTask.from_data(
                    ligand_mol=ligand_rdmol,
                    protein_path=pdb_file,
                    ligand_path=sdf_path,
                )

                #results = task.run_sync(score_only=score_only)
                task.run(score_only=score_only)
                results = task.get_results()
                if results:
                    best_result = results[0]
                    csvwriter.writerow([sdf_file, best_result['affinity'], None])
                else:
                    csvwriter.writerow([sdf_file, None, "No results"])
            except Exception as e:
                # 捕获任务失败并记录错误
                print(f"Error docking {sdf_file}: {e}")
                if task.errors:
                    csvwriter.writerow([sdf_file, None, task.errors])
                else:
                    csvwriter.writerow([sdf_file, None, str(e)])

if __name__ == "__main__":
    sdf_directory = "/public/home/chensn/DL/DiffDec-master/samples_exper_c_2_19/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512/0"#生成的分子
    pdb_file_path = "/public/home/chensn/DL/DiffDec-master/samples_exper_c_2_19/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512/0/pock_.pdb"#靶点
    output_directory = "/public/home/chensn/DL/DiffDec-master/docking_results_2_19"
    output_csv_path = os.path.join(output_directory, "results.csv")

    dock_all_sdfs_to_pdb(
        sdf_dir=sdf_directory,
        pdb_file=pdb_file_path,
        output_dir=output_directory,
        output_csv=output_csv_path,
        score_only=True  # Enable score_only mode
    )