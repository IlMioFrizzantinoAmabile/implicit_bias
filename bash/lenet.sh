#!/bin/bash
#BSUB -J lenet
#BSUB -q p1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 72:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/gelu_lenet%J.out
#BSUB -e logs/gelu_lenet%J.err

source ./../uncertainty_quantification/virtualenv/bin/activate

act_fun="tanh"

for seed in {1..5}; do

    python train_model.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --n_epochs 0 --run_name "${act_fun}"_WU

    python train_and_measure_gradient.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --batch_size 45000 --optimizer sgd --momentum 0 --test_every_n_epoch 1 --n_epochs 20 --freeze_batchstats --run_name "${act_fun}"_gd
    python train_and_measure_gradient_stoch.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --optimizer sgd --mc_samples 1000 --momentum 0 --test_every_n_epoch 1 --n_epochs 10 --freeze_batchstats --run_name_pretrained "${act_fun}"_WU --run_name "${act_fun}"_gd_fullstats
    # python train_and_measure_gradient.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --batch_size 45000 --optimizer sgd --measure_hessian_trace --mc_samples 1000 --momentum 0 --test_every_n_epoch 1 --n_epochs 20 --freeze_batchstats --run_name_pretrained "${act_fun}"_WU --run_name "${act_fun}"_gd_hess

    for batch_size in 22500 9000 4500 450 45; do
        python train_and_measure_gradient.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --batch_size $batch_size --optimizer sgd --momentum 0 --test_every_n_epoch 1 --n_epochs 20 --freeze_batchstats --run_name "${act_fun}"_sgd_"${batch_size}"
    done

    python train_and_measure_gradient.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --batch_size 45000 --optimizer sam_sgd --momentum 0 --test_every_n_epoch 1 --n_epochs 20 --freeze_batchstats --run_name "${act_fun}"_sam
    python train_and_measure_gradient.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --batch_size 45000 --optimizer iflooding --momentum 0 --test_every_n_epoch 1 --n_epochs 20 --freeze_batchstats --run_name "${act_fun}"_flooding
    # python train_and_measure_gradient.py --activation_fun $act_fun --seed $seed --dataset CIFAR-10 --model LeNet --batch_size 45000 --optimizer noisy_sgd --measure_hessian_trace --mc_samples 1000 --momentum 0 --test_every_n_epoch 1 --n_epochs 10 --freeze_batchstats --run_name "${act_fun}"_noisy_sdg
done