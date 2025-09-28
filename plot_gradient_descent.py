import pickle
import os
import matplotlib.pyplot as plt  

save_path = "../models"
model_name = "MLP_depth1_hidden50"
# model_name = "LeNet"
# model_name = "ResNet"
dataset_name = "CIFAR-10"
act_fn = "tanh"

parameters = []

b = 1
seed = 1
lrs = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
if model_name=="ResNet":
    lrs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    lrs = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    if act_fn == "tanh":
        lrs = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
if model_name=="LeNet":
    lrs = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    lrs = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    lrs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]


#, 8, 16]
# epochs = [1]

# plt.figure()

for seed in range(1,2):

    fig, axs = plt.subplots(3, figsize=(5, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1, 3]})


    cs = [None, 0.02, 0.01, 0.002]
    if model_name=="LeNet":
        cs = [None, 0.1, 0.02, 0.01, 0.002]

    for i,c in enumerate(cs):
        run_name = f"{act_fn}_WU"
        stats_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_stats.pickle"
        stats_dict = pickle.load(open(stats_file_path, 'rb'))
        print(stats_dict)
        base_model_stats_dict = {k: stats_dict[k][-1] for k in stats_dict.keys() if len(stats_dict[k])}


        for l,lr in enumerate(lrs):
    
            run_name = f"{act_fn}_gd_lr{lr}"
            stats_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_stats.pickle"
            stats_dict = pickle.load(open(stats_file_path, 'rb'))

            if c is None:
                print("LR=",lr, "-", len(stats_dict["train_gradient_norm"]), len(stats_dict["valid_gradient_norm"]))

                color = (1-l/len(lrs)) * 0.9 + 0.1
                axs[0].plot([base_model_stats_dict["train_acc_or_mse"]]+stats_dict["train_acc_or_mse"], color=plt.cm.Blues(color))
                axs[1].plot([base_model_stats_dict["train_loss"]]+stats_dict["train_loss"], color=plt.cm.Blues(color))
                axs[2].plot([base_model_stats_dict["train_gradient_norm"]]+stats_dict["train_gradient_norm"], color=plt.cm.Blues(color))
                if l <3:
                    axs[0].plot([base_model_stats_dict["train_acc_or_mse"]]+stats_dict["train_acc_or_mse"], label=f"lr={lr}", color=plt.cm.Blues(color))
            else:
                e = int(c / lr)
                if lr==0.0001 and c==0.002:
                    print(e != c // lr, e, c // lr)
                # if e != c // lr:
                #     continue
                # if not np.isclose(e , (c+1e-7) // lr):
                #     print(e != c // lr, e, (c+1e-7) // lr)#, c, lr)
                #     continue
                color = f"C{i}"
                if e>0 and e<=len(stats_dict["train_loss"]):
                    axs[0].scatter(e, stats_dict["train_acc_or_mse"][e-1], color=color)
                    axs[1].scatter(e, stats_dict["train_loss"][e-1], color=color)
                    axs[2].scatter(e, stats_dict["train_gradient_norm"][e-1], color=color)
                if e==len(stats_dict["train_loss"]):
                    axs[2].plot(e, stats_dict["train_gradient_norm"][e-1], marker='.', color=color, label=f"c={c}")
        # axs[0].scatter(0, base_model_stats_dict["train_acc_or_mse"], color=color)
        # axs[1].scatter(0, base_model_stats_dict["train_loss"], color=color)
        # axs[2].scatter(0, base_model_stats_dict["train_gradient_norm"], color=color, label=f"c={c}")



    axs[0].set_ylabel("accuracy")
    axs[1].set_ylabel("loss")
    axs[2].set_ylabel("gradient norm")

    # axs[0].set_ylim([base_model_stats_dict["train_acc_or_mse"]-0.01, 0.15])
    # if model_name=="ResNet":
    #     axs[1].set_ylim([2.2, base_model_stats_dict["train_loss"]+0.01])
    #     axs[2].set_ylim([1, base_model_stats_dict["train_gradient_norm"]+0.01])
    # if model_name=="LeNet":
    #     axs[1].set_ylim([2.1, base_model_stats_dict["train_loss"]+0.01])
    #     axs[2].set_ylim([0.11, base_model_stats_dict["train_gradient_norm"]+0.005])

    axs[0].set_title(model_name)
    axs[2].set_xlabel("epoch")
    axs[2].set_xticks(list(range(len(stats_dict["train_loss"])+1)))
    axs[2].set_yscale("log")


    for ax in axs:
        ax.grid(True, color='grey', linestyle=':')

    # axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[0].legend()
    axs[2].legend()
    plt.tight_layout()

    os.makedirs(f"figures/seed_{seed}", exist_ok=True)
    plt.savefig(f"figures/seed_{seed}/{act_fn}_{model_name}.png", bbox_inches='tight')
