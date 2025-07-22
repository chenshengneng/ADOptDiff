CUDA_VISIBLE_DEVICES=2 python -W ignore sample.py \
                 --checkpoint ckpt/try.ckpt\
                 --samples sample_mols \
                 --data /public/home/chensn/DL/ADOptDiff/data/data \
                 --prefix bingdingnet_test_full \
                 --n_samples 100 \
                 --device cuda:0
