#pre training
python train.py selfProb \
    -project selfProb \
    -dataset mini_imagenet \
    -dataroot dataSet/MINI_DATA_DIR \
    -base_mode 'ft_dot' \
    -new_mode 'avg_cos' \
    -gamma 0.1  -lr_base 0.1 \
    -decay 0.0005  -epochs_base 400 \
    -schedule Cosine \
    -tmax 400  -gpu '1' \
    -temperature 16 \
    -batch_size_base 128


# Fine Tuning and Testing for miniImageNet

python train.py NEWR  -project NEWR -dataset mini_imagenet -dataroot dataSet/MINI_DATA_DIR  -shift_weight 0.1 -trainTest 'test' -lr_base 0.0001 -decay 0.0005 -schedule Cosine -tmax 20 -epochs_new 25  -NEWR 1 -taskInfo 0 -model_dir 'checkpoint/mini_imagenet/NEWR/ft_cos-avg_cos-data_init-start_0/0213-19-53-50-432-Epo_400-Bs_128-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_400-NormF-T_16.00-tw_16-0.5-soft_proto/session0_max_acc.pth'



