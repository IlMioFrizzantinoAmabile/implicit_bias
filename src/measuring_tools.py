import jax
import jax.numpy as jnp
from jax import flatten_util
import tqdm

from src.models import compute_num_params



def get_dataset_gradient_norm(params_dict, loader, get_gradient):
    if len(loader)==0:
        return 0.
    for b, batch in enumerate(loader):
        X = jnp.array(batch[0].numpy())
        Y = jnp.array(batch[1].numpy())

        gradient = get_gradient(params_dict, X, Y)
        gradient = flatten_util.ravel_pytree(gradient)[0]
        dataset_gradient = gradient if b==0 else dataset_gradient + gradient
    dataset_gradient /= len(loader)
    grad_norm = jnp.sum(dataset_gradient**2)
    return grad_norm.item()




def get_batch_gradient_scalar_product(params_dict, loader, get_gradient):
    if len(loader)<2:
        return 0.
    gradients = []
    for batch in loader:
        X = jnp.array(batch[0].numpy())
        Y = jnp.array(batch[1].numpy())

        gradient = get_gradient(params_dict, X, Y)
        gradient = flatten_util.ravel_pytree(gradient)[0]
        gradients.append(gradient)
        
    gradients = jnp.stack(gradients)
    scalar_products = jnp.einsum("ij,kj->ik", gradients, gradients)
    n_batches = scalar_products.shape[0]
    scalar_products = scalar_products * (1 - jnp.eye(n_batches))
    average_scalar_product = scalar_products.sum() / (n_batches * (n_batches-1))
    return average_scalar_product.item()












def hutchinson(
    matrix_vector_product,
    dim,
    sequential = True,
    n_samples: int = None,
    key: jax.random.PRNGKey = None,
    estimator = "Rademacher",
):
    if n_samples is not None and n_samples >= dim:
        print(f"You called stochastich Hutchinson on dim {dim} with {n_samples} samples, you dumb. I'm switching to deterministic, so it's faster and accurate")
        n_samples = None

    if n_samples is None:
        # deterministic
        n_samples = dim
        base_vectors = jnp.concatenate((jnp.zeros((dim-1,)), jnp.ones((1,)), jnp.zeros((dim-1,))))
        get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (dim-k-1,), (dim,))
        aggr_fun = (lambda x, y : x + y) if sequential else jax.numpy.sum
    else:
        #stochastic
        if key is None:
            raise ValueError("Stochastic estimator needs a random key")
        keys = jax.random.split(key, n_samples)
        if estimator == "Rademacher":
            get_vec = lambda k: jax.random.bernoulli(keys[k], p=0.5, shape=(dim,)).astype(float) * 2 - 1
        elif estimator == "Normal":
            get_vec = lambda k: jax.random.normal(keys[k], shape=(dim,))
        aggr_fun = (lambda x, y : x + y/n_samples) if sequential else jax.numpy.mean

    @jax.jit
    def one_sample_trace(k):
        vec = get_vec(k)
        M_vec = matrix_vector_product(vec)
        trace = vec @ M_vec
        return trace

    if sequential:
        return jax.lax.fori_loop(
            0, n_samples, 
            lambda k, temp : aggr_fun(temp, one_sample_trace(k)), 
            0.
        )
    else:
        traces = jax.vmap(one_sample_trace)(jnp.arange(n_samples)) 
        return aggr_fun(traces, axis=0) 


def get_hessian_trace(
        params_dict, 
        loader, 
        get_hessian_vp,
        n_samples = 1000,
        seed = 0
    ):
    if len(loader)==0:
        return 0.
    for b, batch in enumerate(loader):
        X = jnp.array(batch[0].numpy())
        Y = jnp.array(batch[1].numpy())

        P = compute_num_params(params_dict["params"])
        hessian_vp = get_hessian_vp(params_dict, X, Y)
        vec = jax.random.normal(jax.random.PRNGKey(0), shape=(P,))
        vec = hessian_vp(vec)
        trace = hutchinson(
            hessian_vp,
            P,
            sequential = True,
            n_samples = n_samples,
            key = jax.random.PRNGKey(seed),
            estimator = "Rademacher",
        )
        dataset_trace = trace if b==0 else dataset_trace + trace
    return dataset_trace.item()
