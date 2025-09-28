


    # for epoch in 1 2 4 8 16; do
    #     for lr in 1 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001; do
    #         python train_model.py --seed $seed --dataset CIFAR-10 --model LeNet --batch_size 45000 --optimizer sgd --momentum 0 --test_every_n_epoch $epoch --n_epochs 16 --freeze_batchstats --run_name_pretrained e10 --learning_rate $lr --run_name e10_gd_lr"$lr"_e"$epoch"_b1
    #     done