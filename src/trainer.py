import jax
import jax.numpy as jnp
import torch
import optax
import time
import tqdm

from src.models import compute_num_params, compute_norm_params, Model
from src.losses import get_loss_function
from src.optimizers import sam_mom, sam_adam, iflooding_sgd, sgd_noisy, sgd_noisy_sample_parameters, sgd_noisy_accumulate_gradients


from src.autodiff import initialize_get_gradient, initialize_get_hessian_vp
from src.measuring_tools import get_dataset_gradient_norm, get_batch_gradient_scalar_product, get_hessian_trace



def gradient_descent(
    model: Model,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    args_dict: dict,
    pretrained_params_dict: dict = None,
    opt_state = None,
    extra_loaders = None
):
    """
    Mimimize the loss for a given model and dataset.
    :param model: initialized model to use for training
    :param train_loader: train dataloader (torch.utils.data.DataLoader)
    :param valid_loader: test dataloader (torch.utils.data.DataLoader)
    :param key: random.PRNGKey for jax modules
    :param args_dict: dictionary of arguments for training passed from the command line
    :return: params
    """
    #################
    # observe datas #
    if extra_loaders is not None:
        train_loader_extra, valid_loader_extra = extra_loaders
    else:
        train_loader_extra, valid_loader_extra = None, None

    print(f"There are {len(train_loader) } batches every epoch")
    batch = next(iter(train_loader))
    x_init, y_init = jnp.array(batch[0].numpy()), jnp.array(batch[1].numpy())
    print(f"First batch shape: data = {x_init.shape}, target = {y_init.shape}")

    ##############
    # init model #
    key = jax.random.PRNGKey(args_dict["seed"])
    if model.has_dropout:
        key, key_dropout = jax.random.split(key, 2)
    if pretrained_params_dict is None:
        params_dict = model.init(key, x_init)
        print(params_dict.keys())
    else:
        print("Loading pretrained model")
        params_dict = pretrained_params_dict
    print(f"Model has {compute_num_params(params_dict['params'])} parameters")



    ##################
    # init optimizer #
    optimizer_hparams = args_dict["opt_hp"]
    # learning rate schedule
    if not args_dict["decrease_learning_rate"]:
        lr_schedule = optimizer_hparams.pop('lr')
    else:
        if not args_dict["decrease_learning_rate_piecewise"]:
            #lr_schedule = optax.schedules.warmup_exponential_decay_schedule(
            lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
                init_value = 0., 
                peak_value = args_dict["learning_rate"], 
                warmup_steps = 1 * len(train_loader),    # 1 warmup epoch
                decay_steps = args_dict["n_epochs"] * len(train_loader),
                end_value = 0.,
            )
        else:
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=optimizer_hparams.pop('lr'),
                boundaries_and_scales={
                    int(len(train_loader)*args_dict["n_epochs"]*0.5): 0.1, #int(len(train_loader)*args_dict["n_epochs"]*0.6): 0.1,
                    int(len(train_loader)*args_dict["n_epochs"]*0.75): 0.1 #int(len(train_loader)*args_dict["n_epochs"]*0.85): 0.1
                }
            )

    # define optimizer
    if args_dict["optimizer"] == "sgd":
        if args_dict['weight_decay'] is None:
            optimizer = optax.sgd(
                lr_schedule,
                momentum = args_dict["momentum"]
            )
        else:
            optimizer = optax.chain(
                optax.add_decayed_weights(args_dict['weight_decay']),
                optax.sgd(
                    lr_schedule,
                    momentum = args_dict["momentum"]
                )
            )
    elif args_dict["optimizer"] == "adam":
        optimizer = optax.adam(
            lr_schedule,
            b1 = args_dict["momentum"],
            b2 = args_dict["momentum_hessian"],
            eps = args_dict["adam_eps"]
        )
    elif args_dict["optimizer"] == "adamax":
        optimizer = optax.adamax(
            lr_schedule,
            b1 = args_dict["momentum"],
            b2 = args_dict["momentum_hessian"],
            eps = args_dict["adam_eps"]
        )
    elif args_dict["optimizer"] == "adamw":
        optimizer = optax.adamw(
            lr_schedule,
            b1 = args_dict["momentum"],
            b2 = args_dict["momentum_hessian"],
            eps = args_dict["adam_eps"],
            weight_decay = args_dict["weight_decay"]
        )
    elif args_dict["optimizer"] == "sam_sgd":
        optimizer = sam_mom(
            lr_schedule,
            momentum = args_dict['momentum'],
            rho = lr_schedule, #args_dict['sam_rho'],    # equate learning rates of the two steps
            weight_decay = args_dict["weight_decay"]
        )
    elif args_dict["optimizer"] == "sam_adam":
        optimizer = sam_adam(
            lr_schedule,
            b1 = args_dict['momentum'],
            b2 = args_dict['momentum_hessian'],
            rho = lr_schedule, #args_dict['sam_rho'],    # equate learning rates of the two steps
            eps = args_dict["adam_eps"],
            weight_decay = args_dict["weight_decay"]
        )
    elif args_dict["optimizer"] == "iflooding":
        optimizer = iflooding_sgd(
            lr_schedule,
            momentum = args_dict['momentum'],
            rho = lr_schedule, #args_dict['sam_rho'],    # equate learning rates of the two steps
            hessian_momentum = args_dict['momentum_hessian'],
            weight_decay = args_dict["weight_decay"]
        )
    elif args_dict["optimizer"] == "noisy_sgd":
        optimizer = sgd_noisy(
            lr_schedule,
            ess = 0.1 / lr_schedule, #args_dict["effective_sample_size"], #make noise norm 0.1*lr
            beta1 = args_dict['momentum'],
            weight_decay = args_dict['weight_decay']
        )
        mc_samples = args_dict["mc_samples"]
        key, key_sampling = jax.random.split(key, 2)
        sample_parameters = sgd_noisy_sample_parameters
        accumulate_gradients = sgd_noisy_accumulate_gradients
        def get_sample_gradient_and_accumulate(loss_fn, param_mode):
            def sample_gradient_and_accumulate(opt_state, k):
                param_sample, opt_state = sample_parameters(
                    k, param_mode, opt_state
                )
                _, grads = jax.value_and_grad(loss_fn, has_aux=True)(param_sample)
                opt_state = accumulate_gradients(grads, opt_state)
                return opt_state, None
            return sample_gradient_and_accumulate
    else:
        raise ValueError(f"What do you mean with '{args_dict['optimizer']}' optimizer?")
    
    if args_dict["optimizer"] in ["sgd","adam", "adamax", "adamw"]:
        opt_type = 'classic'
    elif args_dict["optimizer"] in ["sam_sgd", "sam_adam", "iflooding"]:
        opt_type = 'half_step'
    elif args_dict["optimizer"] in ["noisy_sgd"]:
        opt_type = 'variational'

    if opt_state is None:
        opt_state = optimizer.init(params_dict['params'])
    


    #############
    # init loss 
    loss_function_train, loss_function_test = get_loss_function(
        model,
        likelihood = args_dict["likelihood"],
        class_frequencies = train_loader.dataset.dataset.class_frequencies if args_dict["likelihood"]=="binary_multiclassification" else None
    )
    
    ########################
    # define training step #
    if not model.has_batch_stats:
        #####################
        #### MLP, LeNet #####
        # print("Training MLP or LeNet")
        if opt_type == 'classic':
            @jax.jit
            def train_step(opt_state, params_dict, x, y):
                loss_fn = lambda p: loss_function_train(p, x, y)
                # Get loss, gradients for loss, and other outputs of loss function
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, ) = ret
                # Update parameters
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
                return opt_state, params_dict, loss, acc_or_sse
        elif opt_type == 'half_step':
            @jax.jit
            def train_step(opt_state, params_dict, x, y):
                loss_fn = lambda p: loss_function_train(p, x, y)
                # first half step
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, ) = ret
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
                # second half step
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, ) = ret
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
                return opt_state, params_dict, loss, acc_or_sse
        elif opt_type == 'variational':
            @jax.jit
            def train_step(opt_state, params_dict, x, y, key_s):
                key_s, key_s_this_step = jax.random.split(key_s, 2)
                ks = jax.random.split(key_s_this_step, mc_samples)
                    
                loss_fn = lambda p: loss_function_train(p, x, y)
                sample_gradient_and_accumulate = get_sample_gradient_and_accumulate(loss_fn, params_dict["params"])
                opt_state, _ = jax.lax.scan(sample_gradient_and_accumulate, opt_state, ks)
                
                # ivon optimizer.update does not use grads, since all the information is stored in opt_state
                param_updates, opt_state = optimizer.update(None, opt_state, params_dict['params'])
                params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
                    
                loss, (acc_or_sse, ) = loss_function_train(params_dict["params"], x, y)
                return opt_state, params_dict, loss, acc_or_sse, key_s
    elif model.has_batch_stats:
        if not args_dict["freeze_batchstats"]:
            #################
            #### ResNet #####
            # print("Training ReNet")
            if opt_type == 'classic':
                @jax.jit
                def train_step(opt_state, params_dict, x, y):
                    loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y)
                    # Get loss, gradients for loss, and other outputs of loss function
                    ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                    loss, (acc_or_sse, new_model_state) = ret
                    # Update parameters and batch statistics
                    param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                    params_dict = {
                        'params' : optax.apply_updates(params_dict['params'], param_updates),
                        'batch_stats' : new_model_state['batch_stats']
                    }
                    return opt_state, params_dict, loss, acc_or_sse
            elif opt_type == 'half_step':
                @jax.jit
                def train_step(opt_state, params_dict, x, y):
                    loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y)

                    # first half step
                    # Get loss, gradients for loss, and other outputs of loss function
                    ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                    loss, (acc_or_sse, new_model_state) = ret
                    # Update parameters and batch statistics
                    param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                    params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)

                    #second half step
                    _, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                    # Update parameters and batch statistics
                    param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                    params_dict = {
                        'params' : optax.apply_updates(params_dict['params'], param_updates),
                        'batch_stats' : new_model_state['batch_stats'] # this comes from the first half step
                    }
                    return opt_state, params_dict, loss, acc_or_sse
            elif opt_type == 'variational':
                @jax.jit
                def train_step(opt_state, params_dict, x, y, key_s):
                    key_s, key_s_this_step = jax.random.split(key_s, 2)
                    ks = jax.random.split(key_s_this_step, mc_samples)

                    loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y)
                    sample_gradient_and_accumulate = get_sample_gradient_and_accumulate(loss_fn, params_dict["params"])
                    opt_state, _ = jax.lax.scan(sample_gradient_and_accumulate, opt_state, ks)

                    param_updates, opt_state = optimizer.update(None, opt_state, params_dict['params'])
                    loss, (acc_or_sse, new_model_state) = loss_fn(params_dict['params'])
                    params_dict = {
                        'params' : optax.apply_updates(params_dict['params'], param_updates),
                        'batch_stats' : new_model_state['batch_stats']
                    }
                    return opt_state, params_dict, loss, acc_or_sse, key_s
        else:
            #################
            #### ResNet #####
            # freezed batchstats
            if opt_type == 'classic':
                @jax.jit
                def train_step(opt_state, params_dict, x, y):
                    loss_fn = lambda p: loss_function_test(p, params_dict['batch_stats'], x, y)  
                    # Get loss, gradients for loss, and other outputs of loss function
                    ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                    loss, (acc_or_sse, new_model_state) = ret
                    # Update parameters and batch statistics
                    param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                    params_dict = {
                        'params' : optax.apply_updates(params_dict['params'], param_updates),
                        'batch_stats' : params_dict['batch_stats']                                
                    }
                    return opt_state, params_dict, loss, acc_or_sse
            elif opt_type == 'half_step':
                @jax.jit
                def train_step(opt_state, params_dict, x, y):
                    loss_fn = lambda p: loss_function_test(p, params_dict['batch_stats'], x, y)   

                    # first half step
                    # Get loss, gradients for loss, and other outputs of loss function
                    ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                    loss, (acc_or_sse, new_model_state) = ret
                    # Update parameters and batch statistics
                    param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                    params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)

                    #second half step
                    _, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                    # Update parameters and batch statistics
                    param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                    params_dict = {
                        'params' : optax.apply_updates(params_dict['params'], param_updates),
                        'batch_stats' : params_dict['batch_stats']      
                    }
                    return opt_state, params_dict, loss, acc_or_sse
            elif opt_type == 'variational':
                @jax.jit
                def train_step(opt_state, params_dict, x, y, key_s):
                    key_s, key_s_this_step = jax.random.split(key_s, 2)
                    ks = jax.random.split(key_s_this_step, mc_samples)

                    loss_fn = lambda p: loss_function_test(p, params_dict['batch_stats'], x, y)  
                    sample_gradient_and_accumulate = get_sample_gradient_and_accumulate(loss_fn, params_dict["params"])
                    opt_state, _ = jax.lax.scan(sample_gradient_and_accumulate, opt_state, ks)

                    param_updates, opt_state = optimizer.update(None, opt_state, params_dict['params'])
                    loss, (acc_or_sse, new_model_state) = loss_fn(params_dict['params'])
                    params_dict = {
                        'params' : optax.apply_updates(params_dict['params'], param_updates),
                        'batch_stats' : params_dict['batch_stats']                   
                    }
                    return opt_state, params_dict, loss, acc_or_sse, key_s



    get_gradient = initialize_get_gradient(model, loss_function_test)
    if args_dict["measure_hessian_trace"]:
        get_hessian_vp = initialize_get_hessian_vp(model, loss_function_test)

    def get_precise_stats(model, params_dict, loader, extra_loader):
        loss = 0.
        acc_or_sse = 0. if args_dict["likelihood"] != "binary_multiclassification" else jnp.zeros((y_init.shape[1], ))
        start_time = time.time()
        for batch in loader:
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())
            if model.has_batch_stats:
                batch_loss, (batch_acc_or_sse, _) = loss_function_test(
                    params_dict['params'], 
                    params_dict['batch_stats'], 
                    X, 
                    Y
                )
            else:
                batch_loss, (batch_acc_or_sse, ) = loss_function_test(
                    params_dict['params'], 
                    X, 
                    Y
                )
            loss += batch_loss.item()
            acc_or_sse += batch_acc_or_sse/X.shape[0]
        acc_or_sse = acc_or_sse/len(loader) if len(loader)>0 else 0
        loss = loss/len(loader) if len(loader)>0 else 0

        gradient_norm = get_dataset_gradient_norm(params_dict, loader, get_gradient)
        gradient_scalar_product = get_batch_gradient_scalar_product(params_dict, loader, get_gradient)
        if extra_loader is None: 
            gradient_scalar_product_exta = []
            hessian_trace = get_hessian_trace(params_dict, loader, get_hessian_vp, n_samples=1000, seed=args_dict["seed"]+420) if args_dict["measure_hessian_trace"] else 0.   
        else:
            gradient_scalar_product_exta = [get_batch_gradient_scalar_product(params_dict, l, get_gradient) for l in extra_loader]
            hessian_trace = get_hessian_trace(params_dict, extra_loader[-1], get_hessian_vp, n_samples=1000, seed=args_dict["seed"]+420) if args_dict["measure_hessian_trace"] else 0.   
        return loss, acc_or_sse, gradient_norm, gradient_scalar_product, gradient_scalar_product_exta, hessian_trace, time.time() - start_time


    #########################
    #########################
    # start training epochs #
    #########################
    print("Starting training...")
    epoch_stats_dict = { # computed while parameters change 
        "loss": [],
        "acc_or_mse": [],
        "params_norm" : [] }
    train_stats_dict = { # computed with fixed parameters
        "loss": [],
        "acc_or_mse": [],
        "gradient_norm": [],
        "gradient_scalar_product": [],
        "gradient_scalar_product_extra": [],
        "hessian_trace": [] }
    valid_stats_dict = { # computed with fixed parameters
        "loss": [],
        "acc_or_mse": [],
        "gradient_norm": [],
        "gradient_scalar_product": [],
        "gradient_scalar_product_extra": [],
        "hessian_trace": [] }
    for epoch in range(1, args_dict["n_epochs"] + 1):
        loss, acc_or_sse = 0., 0. 
        start_time = time.time()
        train_loader_bar = tqdm.tqdm(train_loader) if args_dict["verbose"] else train_loader
        for batch in train_loader_bar:
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            if opt_type=='variational':
                opt_state, params_dict, batch_loss, batch_acc_or_sse, key_sampling = train_step(opt_state, params_dict, X, Y, key_sampling)
            else:
                opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y)

            loss += batch_loss.item()
            batch_acc_or_sse /= X.shape[0]
            acc_or_sse += batch_acc_or_sse

            if args_dict["verbose"]:
                train_loader_bar.set_description(f"Epoch {epoch}/{args_dict['n_epochs']}, batch loss = {batch_loss.item():.3f}, acc = {batch_acc_or_sse:.3f}")
    
        acc_or_sse /= len(train_loader)
        loss /= len(train_loader)
        params_norm = compute_norm_params(params_dict['params'])
        batch_stats_norm = compute_norm_params(params_dict['batch_stats']) if model.has_batch_stats else 0.
        print(f"epoch={epoch} averages - loss={loss:.3f}, params norm={params_norm:.2f}, batch_stats norm={batch_stats_norm:.2f}, acc={acc_or_sse:.3f}, time={time.time() - start_time:.3f}s")
        epoch_stats_dict["loss"].append(loss)
        epoch_stats_dict["acc_or_mse"].append(acc_or_sse)
        epoch_stats_dict["params_norm"].append(params_norm)


        if epoch % args_dict["test_every_n_epoch"] != 0 and epoch != args_dict["n_epochs"]:
            continue

        
        for label, loader, loader_extra, stats_dict in [("Train", train_loader, train_loader_extra, train_stats_dict), ("Validation", valid_loader, valid_loader_extra, valid_stats_dict)]:
            loss, acc_or_sse, gradient_norm, gradient_scalar_product, gradient_scalar_product_exta, hessian_trace, duration = get_precise_stats(model, params_dict, loader, loader_extra)
            print(f"{label} stats\t - loss={loss:.3f}, acc={acc_or_sse:.3f}, grad_norm={gradient_norm:.3f}, grad_scalar_product={gradient_scalar_product:.3f}, grad_scalar_product_extra={gradient_scalar_product_exta}, time={duration:.3f}s")
            stats_dict["loss"].append(loss)
            stats_dict["acc_or_mse"].append(acc_or_sse)
            stats_dict["gradient_norm"].append(gradient_norm)
            stats_dict["gradient_scalar_product"].append(gradient_scalar_product)
            stats_dict["gradient_scalar_product_extra"].append(gradient_scalar_product_exta)
            stats_dict["hessian_trace"].append(hessian_trace)
    
    if args_dict["n_epochs"]==0:
        for label, loader, loader_extra, stats_dict in [("Train", train_loader, train_loader_extra, train_stats_dict), ("Validation", valid_loader, valid_loader_extra, valid_stats_dict)]:
            loss, acc_or_sse, gradient_norm, gradient_scalar_product, gradient_scalar_product_exta, hessian_trace, duration = get_precise_stats(model, params_dict, loader, loader_extra)
            print(f"{label} stats\t - loss={loss:.3f}, acc={acc_or_sse:.3f}, grad_norm={gradient_norm:.3f}, grad_scalar_product={gradient_scalar_product:.3f}, grad_scalar_product_extra={gradient_scalar_product_exta}, time={duration:.3f}s")
            stats_dict["loss"].append(loss)
            stats_dict["acc_or_mse"].append(acc_or_sse)
            stats_dict["gradient_norm"].append(gradient_norm)
            stats_dict["gradient_scalar_product"].append(gradient_scalar_product)
            stats_dict["gradient_scalar_product_extra"].append(gradient_scalar_product_exta)
            stats_dict["hessian_trace"].append(hessian_trace)

    epoch_stats_dict = {'epoch_'+k : v for k,v in epoch_stats_dict.items()}    
    train_stats_dict = {'train_'+k : v for k,v in train_stats_dict.items()}    
    valid_stats_dict = {'valid_'+k : v for k,v in valid_stats_dict.items()}    
    stats_dict = {**train_stats_dict, **valid_stats_dict, **epoch_stats_dict}

    return params_dict, stats_dict, opt_state