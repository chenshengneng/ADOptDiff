CUDA_VISIBLE_DEVICES=2 python -W ignore sample_multi.py \
                 --checkpoint models/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_seed21_bs8_date18-02_time14-36-01.822568/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_seed21_bs8_date18-02_time14-36-01.822568_best_epoch=epoch=279.ckpt\
                 --samples sample_mols_cla2_softmax_test100_seed21 \
                 --data /public/home/chensn/DL/DiffDec-master/data/multi \
                 --prefix bingdingnet_test_full \
                 --n_samples 100 \
                 --device cuda:0
