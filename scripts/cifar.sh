#pre-training
python train.py selfProb  -project selfProb -dataset cifar100 -dataroot dataSet/CIFAR_DATA_DIR  
-base_mode 'ft_cos'  -new_mode 'avg_cos' -lr_base 0.1 -decay 0.0005  -epochs_base 400 -batch_size_base 256 
 -schedule Cosine -tmax 400  -gpu '0' -temperature 16  -softmax_t 16  -shift_weight 0.1


# Fine Tuning and Testing for CIFAR100

python train.py NEWR  -project NEWR -dataset cifar100 -dataroot dataSet/CIFAR_DATA_DIR -shift_weight 0.1 -trainTest 'test' -lr_base 0.0001 -decay 0.0005 -schedule Cosine -tmax 20 -epochs_new 25  -NEWR 1 -taskInfo 0 -model_dir 'checkpoint/cifar100/NEWR/ft_cos-avg_cos-data_init-start_0/1223-18-20-41-423-Epo_600-Bs_256-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_600-NormF-T_16.00-tw_16.0-0.1-soft_proto/session0_max_acc.pth'

