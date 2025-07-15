import os
import shutil
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# 定义输入和输出路径
input_dir = '/public/home/chensn/DL/DiffDec-master/sample_mols/diffdec_multi'
output_dir = '/public/home/chensn/DL/DiffDec-master/diffdec_multi_good'

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 初始化计数器
valid_count = 0

# 存储所有筛选出来的分子（有效分子）
valid_molecules = []

# 定义检查分子是否为碎片的函数
def is_fragment(molecule):
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    return len(fragments) > 1  # 如果片段数 > 1，则该分子为碎片

# 遍历所有一级子文件夹
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    
    if os.path.isdir(subfolder_path):
        # 存储当前子文件夹的相似性结果
        similarity_data = []
        
        # 找到以数字加_命名的sdf文件和以true_开头的sdf文件
        sdf_files = [f for f in os.listdir(subfolder_path) if f.endswith('.sdf')]
        numbered_files = [f for f in sdf_files if f.split('_')[0].isdigit()]
        true_files = [f for f in sdf_files if f.startswith('true_')]
        
        # 处理符合条件的文件并筛选分子
        for num_file in numbered_files:
            for true_file in true_files:
                # 加载分子
                num_mol = Chem.SDMolSupplier(os.path.join(subfolder_path, num_file))[0]
                true_mol = Chem.SDMolSupplier(os.path.join(subfolder_path, true_file))[0]
                
                if num_mol and true_mol and not is_fragment(num_mol):  # 先筛选符合条件的分子
                    # 将有效分子存储
                    valid_molecules.append(num_mol)
                    
                    # 将符合规则的文件复制到输出文件夹
                    output_subfolder_path = os.path.join(output_dir, subfolder)
                    os.makedirs(output_subfolder_path, exist_ok=True)
                    valid_sdf_path = os.path.join(output_subfolder_path, num_file)
                    shutil.copy(os.path.join(subfolder_path, num_file), valid_sdf_path)
                    valid_count += 1

                    # 计算相似性
                    num_fp = AllChem.GetMorganFingerprint(num_mol, 2)
                    true_fp = AllChem.GetMorganFingerprint(true_mol, 2)
                    similarity = DataStructs.TanimotoSimilarity(num_fp, true_fp)
                    
                    # 添加结果到列表
                    similarity_data.append([num_file, true_file, similarity])

        # 计算相似性结果并保存
        if similarity_data:
            similarity_df = pd.DataFrame(similarity_data, columns=['Numbered_File', 'True_File', 'Similarity'])
            mean_similarity = similarity_df['Similarity'].mean()
            median_similarity = similarity_df['Similarity'].median()

            # 添加平均值和中位数到 DataFrame 中
            similarity_df = pd.concat([
                similarity_df,
                pd.DataFrame([["average", "", mean_similarity], ["median", "", median_similarity]], columns=similarity_df.columns)
            ])
            
            # 保存相似性结果到 CSV 文件
            output_csv = os.path.join(subfolder_path, f"{subfolder}_similarity_scores.csv")
            similarity_df.to_csv(output_csv, index=False)
            print(f"相似性结果已保存到 {output_csv}，包含平均值和中位数")

print(f"筛选出的有效分子总数: {valid_count}")
