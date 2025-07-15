CUDA_VISIBLE_DEVICES=0 python -W ignore sample_multi_for_specific_context.py \
        --scaffold_smiles_file /public/home/chensn/experimen/exper_2_19/scaffold_c.smi \
        --protein_file /public/home/chensn/experimen/exper_2_19/pocket_10A.pdb \
        --scaffold_file /public/home/chensn/experimen/exper_2_19/scaffold_good_c.sdf \
        --task_name exp \
        --data_dir data/examples_exper_c_2_19 \
        --checkpoint /public/home/chensn/DL/DiffDec-master/models/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512.ckpt \
        --samples_dir samples_exper_c_2_19 \
        --n_samples 100 \
        --device cuda:0
