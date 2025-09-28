import jax
import jax.numpy as jnp
from functools import partial
from typing import Literal

from src.models.wrapper import Model

def mse_loss(preds, y):
    residual = preds.reshape(-1) - y.reshape(-1)
    return jnp.mean(residual**2)

@partial(jax.jit, static_argnames=['rho'])
def log_gaussian_log_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, output_dim) (predictions)
    y: (batch_size, output_dim) (targets)
    """
    O = y.shape[-1]
    return 0.5 * O * jnp.log(2 * jnp.pi) - 0.5 * O * jnp.log(rho) + 0.5 * rho * mse_loss(preds, y)

@partial(jax.jit, static_argnames=['rho'])
def cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, n_classes) (logits)
    y: (batch_size, n_classes) (one-hot labels)
    """
    preds = preds * rho
    preds = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.mean(jnp.sum(preds * y, axis=-1))


def get_likelihood(
        likelihood: Literal["classification", "regression"] = "classification",
        class_frequencies = None
    ):

    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
        extra_stats_function = lambda preds, y : jnp.sum((preds-y)**2)                                  # sum of squared error
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
        extra_stats_function = lambda preds, y : jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))    # accuracy
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification', 'binary_multiclassification' or 'perception_loss'.")
    
    return negative_log_likelihood, extra_stats_function


def get_loss_function(
        model: Model,
        likelihood: Literal["classification", "regression", "binary_multiclassification", "perception_loss"] = "classification",
        class_frequencies = None
    ):
    negative_log_likelihood, extra_stats_function = get_likelihood(
        likelihood = likelihood,
        class_frequencies = class_frequencies
    )
    if not model.has_batch_stats:
        # MLP, LeNet
        @jax.jit
        def loss_function_train(params, x, y):
            preds = model.apply_train(params, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, )
        @jax.jit
        def loss_function_test(params, x, y):
            preds = model.apply_test(params, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, )
    else:
        if not model.has_dropout:
            # ResNet
            @jax.jit
            def loss_function_train(params, batch_stats, x, y):
                preds, new_model_state = model.apply_train(params, batch_stats, x)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, new_model_state)
            @jax.jit
            def loss_function_test(params, batch_stats, x, y):
                preds = model.apply_test(params, batch_stats, x)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, None)
        else:
            # VAN
            @jax.jit
            def loss_function_train(params, batch_stats, x, y, key_dropout):
                preds, new_model_state = model.apply_train(params, batch_stats, x, key_dropout)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, new_model_state)
            @jax.jit
            def loss_function_test(params, batch_stats, x, y):
                preds = model.apply_test(params, batch_stats, x)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, None)
    
    return loss_function_train, loss_function_test