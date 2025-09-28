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

lrs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]


#, 8, 16]
# epochs = [1]

# plt.figure()

batch_sizes = [22500, 9000, 4500, 450, 45]
batch_idx = 2
batch_size = batch_sizes[batch_idx]

epochs = [0,1,2,5,9]
epochs = [0]

plt.figure(figsize=(4,4))


for seed in range(1,6):

    for epoch in epochs:
        for batch_idx in range(5):
            batch_size = batch_sizes[batch_idx]


            scalar_prods_gd, scalar_prods_sgd = [], []

            for l,lr in enumerate(lrs):

                run_name = f"{act_fn}_gd_fullstats_lr{lr}"
                stats_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_stats.pickle"
                stats_dict = pickle.load(open(stats_file_path, 'rb'))

                grad_scalar_products_from_GD = [stats[batch_idx] for stats in stats_dict["train_gradient_scalar_product_extra"]]


                run_name = f"{act_fn}_sgd_{batch_size}_lr{lr}"
                stats_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_stats.pickle"
                stats_dict = pickle.load(open(stats_file_path, 'rb'))

                grad_scalar_products_from_SGD = stats_dict["train_gradient_scalar_product"]

                sc_gd = grad_scalar_products_from_GD[epoch]
                sc_sgd = grad_scalar_products_from_SGD[epoch]

                scalar_prods_gd.append(sc_gd)
                scalar_prods_sgd.append(sc_sgd)


            print("gd ",scalar_prods_gd)
            print("sgd", scalar_prods_sgd)
            #print([a-b for a,b in zip(scalar_prods_sgd,scalar_prods_gd)])

            plt.plot(lrs, [a-b for a,b in zip(scalar_prods_sgd,scalar_prods_gd)], label=f"bs {batch_size}")




plt.ylabel("delta gradient scalar products")

plt.title(model_name)
plt.xlabel("lr")
plt.xticks(lrs)
plt.xscale("log")


plt.grid(True, color='grey', linestyle=':')
plt.legend()
plt.tight_layout()

os.makedirs(f"figures/", exist_ok=True)
plt.savefig(f"figures/figure2_{act_fn}_{model_name}.png", bbox_inches='tight')
