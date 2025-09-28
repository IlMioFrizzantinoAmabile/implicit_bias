import pickle
import os
import argparse
import json
import datetime

from src.datasets import dataloader_from_string, get_output_dim
from src.models import model_from_string, pretrained_model_from_string

from src.trainer import gradient_descent


parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "CelebA_unsupervised", "ImageNet", "ImageNet_unsupervised"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--split_train_val_ratio", type=float, default=1.)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet", "ResNet", "VAN_tiny", "VAN_small", "VAN_base", "VAN_large"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu", "gelu"], default="gelu", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of layers in the MLP.")
parser.add_argument("--freeze_batchstats", action="store_true", required=False, default=False)

# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--dont_shuffle_batches", action="store_true", required=False, default=False)
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "adamax", "rmsprop", "sam_sgd", "sam_adam", "isam_sgd", "noisy_sgd", "antithetic_sgd", "ivon", "proj_ivon", "proj_pos_ivon", "antithetic_ivon", "antithetic_proj_ivon", "antithetic_proj_pos_ivon", "ivon_subspace", "iflooding", "vadam"], default="adam")
parser.add_argument("--decrease_learning_rate", action="store_true", required=False, default=False)
parser.add_argument("--decrease_learning_rate_piecewise", action="store_true", required=False, default=False)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--momentum_hessian", type=float, default=0.999)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification", "binary_multiclassification", "perception_loss"], default="classification")
parser.add_argument("--sam_rho", type=float, default=0.1)
parser.add_argument("--adam_eps", type=float, default=1e-8)

parser.add_argument("--measure_hessian_trace", action="store_true", required=False, default=False)
parser.add_argument("--mc_samples", type=int, default=10)


# storage
parser.add_argument("--run_name", type=str, default=None, help="Fix the save file name. If None it's set to starting time")
parser.add_argument("--run_name_pretrained", type=str, default=None, help="Run name from which to load pretrained parameters. If None parameters are randomly initialized")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--test_every_n_epoch", type=int, default=20, help="Frequency of computing validation stats")
parser.add_argument("--store_opt_state", action="store_true", required=False, default=False)
parser.add_argument("--load_pretrained_opt_state", action="store_true", required=False, default=False)

# print more stuff
parser.add_argument("--verbose", action="store_true", required=False, default=False)

#python train_model.py --dataset CIFAR-10 --model ResNet --activation_fun relu --optimizer ivon --n_epochs 20 --learning_rate 0.1 --momentum 0.9 --weight_decay 1e-4 --mc_samples 2 --effective_sample_size 1000 --hess_init 1. --test_every_n_epoch 1 --run_name ivon
#python train_model.py --dataset CIFAR-10 --model ResNet --activation_fun relu --optimizer noisy_sgd --n_epochs 20 --learning_rate 0.1 --momentum 0.9 --weight_decay 1e-4 --mc_samples 2 --effective_sample_size 1000 --test_every_n_epoch 1 --run_name noisy_sgd
#python train_model.py --dataset CIFAR-10 --model ResNet --activation_fun relu --optimizer antithetic_sgd --n_epochs 20 --learning_rate 0.1 --momentum 0.9 --weight_decay 1e-4 --mc_samples 1 --effective_sample_size 1000 --test_every_n_epoch 1 --run_name anti_sgd



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


    #############
    ### model ###
    output_dim = get_output_dim(args.dataset)
    model = model_from_string(
        args.model, 
        output_dim, 
        activation_fun = args_dict["activation_fun"],
        mlp_num_layers = args_dict["mlp_num_layers"],
        mlp_hidden_dim = args_dict["mlp_hidden_dim"],
    )
    args_dict["output_dim"] = output_dim
    args_dict["opt_hp"] = {
            # "lr": args_dict["learning_rate"],
            "momentum": args_dict["momentum"],
            "weight_decay": args_dict["weight_decay"],
        }


    ###############
    ### dataset ###
    train_loaders, test_loaders = [], []
    for batch_size in [45000, 22500, 9000, 4500, 450, 45]:
        args_dict["batch_size"] = batch_size

        train_loader, _, test_loader = dataloader_from_string(
            args.dataset,
            n_samples = args.n_samples,
            batch_size = args.batch_size,
            shuffle = not args.dont_shuffle_batches,
            seed = args.seed,
            download = False,
            data_path = args.data_path,
            split_train_val_ratio = args.split_train_val_ratio
        )
        print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(test_loader.dataset)}")
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    extra_loaders = (train_loaders[1:], test_loaders[1:])
    train_loader, test_loader = train_loaders[0], test_loaders[0]
    print(train_loader, test_loader, extra_loaders)

    ################
    ### training ###  
    for lr in [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        args_dict["learning_rate"] = lr
        args_dict["opt_hp"]["lr"] = lr

        if args.run_name_pretrained is not None:
            _, pretrained_params_dict, _ = pretrained_model_from_string(
                model_name = args.model if args.model!="MLP" else f"MLP_depth{args_dict['mlp_num_layers']}_hidden{args_dict['mlp_hidden_dim']}",
                dataset_name = args.dataset,
                n_samples = args.n_samples,
                run_name = args.run_name_pretrained,
                seed = args.seed,
                save_path = args.model_save_path
            )
            if args.load_pretrained_opt_state:
                opt_state = pretrained_params_dict.pop("opt_state")

        params_dict, stats_dict, opt_state = gradient_descent(
                model, 
                train_loader, 
                test_loader, 
                args_dict,
                pretrained_params_dict = None if args.run_name_pretrained is None else pretrained_params_dict,
                opt_state = None if not args.load_pretrained_opt_state else opt_state,
                extra_loaders = extra_loaders
            )
        model_dict = {"model": args.model, **params_dict}
        if args.store_opt_state:
            model_dict["opt_state"] = opt_state


        ####################################
        ### save params and dictionaries ###
        # first folder is dataset
        save_folder = f"{args.model_save_path}/{args.dataset}"
        if args.n_samples is not None:
            save_folder += f"_samples{args.n_samples}"
        # second folder is model
        if args.model == "MLP":
            save_folder += f"/MLP_depth{args.mlp_num_layers}_hidden{args.mlp_hidden_dim}"
        else:
            save_folder += f"/{args.model}"
        # third folder is seed
        save_folder += f"/seed_{args.seed}"
        os.makedirs(save_folder, exist_ok=True)
        
        if args.run_name is not None:
            save_name = f"{args.run_name}_lr{lr}"
        else:
            save_name = f"started_{now_string}"

        print(f"Saving to {save_folder}/{save_name}")
        pickle.dump(model_dict, open(f"{save_folder}/{save_name}_params.pickle", "wb"))
        pickle.dump(stats_dict, open(f"{save_folder}/{save_name}_stats.pickle", "wb"))
        with open(f"{save_folder}/{save_name}_args.json", "w") as f:
            json.dump(args_dict, f)

        print(stats_dict["train_gradient_norm"])
        print(stats_dict["train_gradient_scalar_product"])