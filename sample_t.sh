CUDA_VISIBLE_DEVICES=0 python -W ignore sample_for_specific_context.py \
        --scaffold_smiles_file /path/to/smiles/of/scaffold\
        --protein_file /path/to/pdb/of/pocket \
        --scaffold_file /path/to/sdf/of/scaffold \
        --task_name exp \
        --data_dir data/examples \
        --checkpoint data/data/try.ckpt \
        --samples_dir samples_exper_c \
        --n_samples 100 \
        --device cuda:0
