import pickle
import os
import matplotlib.pyplot as plt  
import numpy as np

save_path = "../models"
model_name = "MLP_depth1_hidden50"
model_name = "LeNet"
model_name = "ResNet"
dataset_name = "CIFAR-10"
act_fn = "tanh"
act_fn = "gelu"

parameters = []

b = 1
seed = 1
lrs = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
if model_name=="ResNet":
    lrs = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    if act_fn == "tanh":
        lrs = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
if model_name=="LeNet":
    lrs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]



plt.figure(figsize=(4,4))

cs = [0.02, 0.01, 0.002]
# if model_name=="LeNet":
#     cs = [0.1, 0.02, 0.01, 0.002]

all_norms = {c : [] for c in cs} 
all_norms_e = {c : [] for c in cs} 

for seed in range(1,6):

    for i,c in enumerate(cs):
        run_name = f"{act_fn}_WU"
        stats_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_stats.pickle"
        stats_dict = pickle.load(open(stats_file_path, 'rb'))
        base_model_stats_dict = {k: stats_dict[k][-1] for k in stats_dict.keys() if len(stats_dict[k])}

        
        grad_norm, grad_norm_e = [], []

        for l,lr in enumerate(lrs):
    
            run_name = f"{act_fn}_gd_lr{lr}"
            stats_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_stats.pickle"
            stats_dict = pickle.load(open(stats_file_path, 'rb'))
            
            
            e = int(c / lr)
            if not np.isclose(e , (c+1e-7) // lr):
                print(e != c // lr, e, (c+1e-7) // lr)#, c, lr)
                continue
            if e>0 and e<=len(stats_dict["train_loss"]):
                grad_norm_e.append(e)
                grad_norm.append(stats_dict["train_gradient_norm"][e-1])
        print(grad_norm,grad_norm_e)
            
        grad_norm = [g - grad_norm[0] for g in grad_norm]
        all_norms[c].append(grad_norm)
        all_norms_e[c].append(grad_norm_e)

        # color = f"C{i}"
        # if seed-1:
        #     plt.plot(grad_norm_e, grad_norm, color=color, marker='.', alpha=0.5)
        # else:
        #     plt.plot(grad_norm_e, grad_norm, color=color, marker='.', alpha=0.8, label=f"c={c}")

for i,c in enumerate(cs):
    aaa = np.array(all_norms[c])
    bbb = np.array(all_norms_e[c])
    print(aaa.shape,bbb.shape)
    # print(aaa,grad_norm_e)
    color = f"C{i}"
    # plt.plot(bbb.T, aaa.T, color=color, marker='.', alpha=1, label=f"c={c}")
    plt.plot(bbb.mean(axis=0), aaa.mean(axis=0), color=color, marker='.', alpha=1, label=f"c={c}")
    plt.fill_between(bbb.mean(axis=0), aaa.mean(axis=0)+aaa.std(axis=0), aaa.mean(axis=0)-aaa.std(axis=0), color=color, alpha=0.1)


plt.ylabel("delta gradient norm")

# if not pretrained:
#     if model_name=="ResNet":
#         plt.ylim([1, base_model_stats_dict["train_gradient_norm"]+0.01])
#     if model_name=="LeNet":
#         axs[1].set_ylim([2.1, base_model_stats_dict["train_loss"]+0.01])
#         axs[2].set_ylim([0.11, base_model_stats_dict["train_gradient_norm"]+0.005])

plt.title(model_name)
plt.xlabel("epoch")
plt.xticks(list(range(len(stats_dict["train_loss"])+1)))
plt.xscale("log")


plt.grid(True, color='grey', linestyle=':')
plt.legend()
plt.tight_layout()

os.makedirs(f"figures/", exist_ok=True)
plt.savefig(f"figures/figure1_{act_fn}_{model_name}.png", bbox_inches='tight')
