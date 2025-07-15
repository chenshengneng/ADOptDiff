python reformat.py --samples_path /public/home/chensn/DL/DiffDec-master/sample_mols_cla2_softmax_test100_seed21/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_seed21_bs8_date18-02_time14-36-01.822568_best_epoch=epoch=279 \
                    --formatted_path ./formatted_softmax_cla2_test100_seed21 \
                    --true_smiles_path /public/home/chensn/DL/DiffDec-master/data/multi/bingdingnet_test_table.csv

python -W ignore compute_metrics.py \
    /public/home/chensn/DL/DiffDec-master/formatted_softmax_cla2_test100_seed42_epoch512/bingdingnet_test_metric.smi

python -W ignore vina_preprocess.py \
    /public/home/chensn/DL/DiffDec-master/formatted_softmax_cla2_test100_seed42_epoch512/bingdingnet_test_metric.smi \
    /public/home/chensn/DL/DiffDec-master/formatted_softmax_cla2_test100_seed42_epoch512/bingdingnet_test_vina.smi 

python vina_docking.py --test_csv_path /public/home/chensn/DL/DiffDec-master/formatted_softmax_cla2_test100_seed42_epoch512/bingdingnet_test_vina.csv \
                    --results_pred_path formatted_softmax_cla2_test100_seed42_epoch512/result.pt \
                    --results_test_path formatted_softmax_cla2_test100_seed42_epoch512/result_testset.pt