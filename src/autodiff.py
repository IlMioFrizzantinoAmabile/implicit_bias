import jax
import jax.numpy as jnp
from jax import flatten_util


def initialize_get_gradient(model, loss_function_test):
    if not model.has_batch_stats:
        # MLP, LeNet
        @jax.jit
        def get_gradient(params_dict, x, y):
            loss_fn = lambda p: loss_function_test(p, x, y)
            _, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
            flat_grad = flatten_util.ravel_pytree(grads)[0]
            return flat_grad
    elif model.has_batch_stats:
        if not model.has_dropout:
            # ResNet
            @jax.jit
            def get_gradient(params_dict, x, y):
                loss_fn = lambda p: loss_function_test(p, params_dict['batch_stats'], x, y)
                # Get loss, gradients for loss, and other outputs of loss function
                _, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                flat_grad = flatten_util.ravel_pytree(grads)[0]
                return flat_grad
        else:
            # VAN
            @jax.jit
            def get_gradient(params_dict, x, y):
                loss_fn = lambda p: loss_function_test(p, params_dict['batch_stats'], x, y)
                # Get loss, gradients for loss, and other outputs of loss function
                _, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                flat_grad = flatten_util.ravel_pytree(grads)[0]
                return flat_grad
    return get_gradient


def initialize_get_hessian_vp(model, loss_function_test):
    if not model.has_batch_stats:
        # MLP, LeNet
        # @jax.jit
        def get_hessian_vp(params_dict, x, y):
            def loss_fn(p): 
                loss, _ = loss_function_test(p, x, y)
                return loss

            @jax.jit
            def hessian_tree_product(tree):
                return jax.jvp(jax.jacrev(loss_fn), (params_dict['params'],), (tree,))[1]
            
            devectorize_fun = flatten_util.ravel_pytree(params_dict['params'])[1]
            @jax.jit
            def hessian_vector_product(v):
                tree = devectorize_fun(v)
                hessian_tree = hessian_tree_product(tree)
                hessian_v = jax.flatten_util.ravel_pytree(hessian_tree)[0]
                return jnp.array(hessian_v)
    
            return hessian_vector_product
    elif model.has_batch_stats:
        # ResNet
        # @jax.jit
        def get_hessian_vp(params_dict, x, y):
            loss_fn = lambda p: loss_function_test(p, params_dict['batch_stats'], x, y)

            @jax.jit
            def hessian_tree_product(tree):
                return jax.jvp(jax.jacrev(loss_fn), (params_dict['params'],), (tree,))[1]
            
            devectorize_fun = flatten_util.ravel_pytree(params_dict['params'])[1]
            @jax.jit
            def hessian_vector_product(v):
                tree = devectorize_fun(v)
                hessian_tree = hessian_tree_product(tree)
                hessian_v = jax.flatten_util.ravel_pytree(hessian_tree)[0]
                return jnp.array(hessian_v)
    
            return hessian_vector_product
    return get_hessian_vp