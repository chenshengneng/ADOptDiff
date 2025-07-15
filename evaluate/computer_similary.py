import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# 定义主文件夹路径
input_dir = '/public/home/chensn/DL/DiffDec-master/filtered_mols'

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
        
        # 计算相似性
        for num_file in numbered_files:
            for true_file in true_files:
                # 加载分子
                num_mol = Chem.SDMolSupplier(os.path.join(subfolder_path, num_file))[0]
                true_mol = Chem.SDMolSupplier(os.path.join(subfolder_path, true_file))[0]
                
                # 检查分子是否加载成功
                if num_mol and true_mol:
                    # 计算指纹和相似性
                    num_fp = AllChem.GetMorganFingerprint(num_mol, 2)
                    true_fp = AllChem.GetMorganFingerprint(true_mol, 2)
                    similarity = DataStructs.TanimotoSimilarity(num_fp, true_fp)
                    
                    # 添加结果到列表
                    similarity_data.append([num_file, true_file, similarity])

        # 计算平均值和中位数
        if similarity_data:
            similarity_df = pd.DataFrame(similarity_data, columns=['Numbered_File', 'True_File', 'Similarity'])
            mean_similarity = similarity_df['Similarity'].mean()
            median_similarity = similarity_df['Similarity'].median()

            # 添加平均值和中位数到 DataFrame 中
            similarity_df = pd.concat([
                similarity_df,
                pd.DataFrame([["avarage", "", mean_similarity], ["mide", "", median_similarity]], columns=similarity_df.columns)
            ])
            
            # 保存 CSV 文件
            output_csv = os.path.join(subfolder_path, f"{subfolder}_similarity_scores.csv")
            similarity_df.to_csv(output_csv, index=False)
            print(f"相似性结果已保存到 {output_csv}，包含平均值和中位数")
