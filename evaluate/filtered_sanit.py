import os
import shutil
from rdkit import Chem

# 定义输入和输出路径
input_dir = '/public/home/chensn/DL/DiffDec-master/sample_mols/multi_chensn_diffdec_multi_bs16_date05-11_time16-19-49.155211_epoch=999'
output_dir = '/public/home/chensn/DL/DiffDec-master/filtered_mols'

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 初始化计数器
valid_count = 0

# 定义检查分子是否为碎片的函数
def is_fragment(molecule):
    # 使用 GetMolFrags 检查分子片段数量
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    return len(fragments) > 1  # 如果片段数 > 1，则该分子为碎片

# 遍历所有一级子文件夹
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    
    # 检查是否为文件夹
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹中的所有文件
        for file_name in os.listdir(subfolder_path):
            # 检查文件是否为SDF文件
            if file_name.endswith('.sdf'):
                sdf_path = os.path.join(subfolder_path, file_name)
                
                # 读取SDF文件并检查化学规则
                suppl = Chem.SDMolSupplier(sdf_path)
                found_valid_molecule = False
                for mol in suppl:
                    if mol is not None and not is_fragment(mol):  # 符合化学规则且不是碎片
                        valid_count += 1
                        
                        # 在输出文件夹中创建对应的子文件夹结构
                        output_subfolder_path = os.path.join(output_dir, subfolder)
                        os.makedirs(output_subfolder_path, exist_ok=True)
                        
                        # 将符合规则的文件复制到对应的输出子文件夹
                        valid_sdf_path = os.path.join(output_subfolder_path, file_name)
                        shutil.copy(sdf_path, valid_sdf_path)
                        found_valid_molecule = True
                        break  # 一旦找到符合规则的分子，就跳出循环
                
                if not found_valid_molecule:
                    print(f"警告：文件 {sdf_path} 中没有符合条件的完整分子")

print(f"符合化学规则且非碎片的SDF文件数量: {valid_count}")
