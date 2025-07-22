python reformat.py --samples_path sample_mol \
                    --formatted_path ./formatted\
                    --true_smiles_path data/data/bingdingnet_test_table.csv

python -W ignore compute_metrics.py \
    ./formatted/bingdingnet_test_metric.smi

python -W ignore vina_preprocess.py \
    ./formatted/bingdingnet_test_metric.smi \
    ./formatted/bingdingnet_test_vina.smi 

python vina_docking.py --test_csv_path ./formatted/bingdingnet_test_vina.csv \
                    --results_pred_path ./formatted/result.pt \
                    --results_test_path ./formatted/result_testset.pt
