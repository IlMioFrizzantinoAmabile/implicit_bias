from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax


def sam_mom(
    lr = 1e-3, 
    momentum = 0.1, 
    rho = 0.1, 
    weight_decay = None,
    sync_period = 2
) -> optax.GradientTransformation:
    """A SAM optimizer using SGD for the outer optimizer."""
    if weight_decay is None:
        opt = optax.sgd(lr, momentum=momentum)
    else:
        opt = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(lr, momentum=momentum)
        )
    adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(rho))
    return optax.contrib.sam(opt, adv_opt, sync_period=sync_period)

def sam_adam(
    lr = 1e-3, 
    b1 = 0.9, 
    b2 = 0.999, 
    rho = 0.1,
    eps = 1e-8,
    weight_decay = None,
    sync_period = 2
) -> optax.GradientTransformation:
    """A SAM optimizer using Adam for the outer optimizer."""
    b1 = 0. if b1 is None else b1
    if weight_decay is None:
        opt = optax.adam(lr, b1=b1, b2=b2, eps=eps)
    else:
        opt = optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)
    adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(rho))
    return optax.contrib.sam(opt, adv_opt, sync_period=sync_period)



def iflooding_sgd(
    lr = 1e-3, 
    momentum = 0.1, 
    rho = 1,
    hessian_momentum = 0.999,
    weight_decay = None
) -> optax.GradientTransformation:
    """A Flooding optimizer using SGD for the outer optimizer."""
    if weight_decay is None:
        opt = optax.sgd(lr, momentum=momentum)
    else:
        opt = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(lr, momentum=momentum)
        )
    return iflooding(opt, rho=rho, hessian_momentum=hessian_momentum)


class iFloodingState(NamedTuple):
  steps_since_sync: jax.Array
  opt_state: optax.OptState
  adv_state: optax.OptState
  cache: optax.Params
  hessian: optax.Updates


def iflooding(
    optimizer: optax.GradientTransformation,
    rho: float = 1,
    hessian_momentum: float = 0.999
) -> optax.GradientTransformation:

  def init_fn(params: optax.Params) -> iFloodingState:
    return iFloodingState(
        steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
        opt_state=optimizer.init(params),
        adv_state=optimizer.init(params),
        cache=params,
        hessian=jax.tree.map(jnp.ones_like, params)
    )

  def pick_one(cond, if_true, if_false):
    return jax.tree.map(
        lambda if_t, if_f: cond * if_t + (1 - cond) * if_f, 
        if_true, 
        if_false
    )

  def update_fn(
      updates: optax.Updates,
      state: iFloodingState,
      params: optax.Params | None = None,
  ) -> tuple[optax.Updates, iFloodingState]:
    first_step = state.steps_since_sync == 0
    last_step = state.steps_since_sync == 1

    adv_updates, adv_state = optimizer.update(
        updates, state.adv_state, params
    )
    adv_updates = jax.tree.map(lambda x: -x, adv_updates)
    adv_hessian = jax.tree.map(
        lambda h, delta, g: hessian_momentum * h + (1.0-hessian_momentum) * 0.5 * (-0.5 * delta) * g / rho**2, 
        state.hessian,
        adv_updates,
        updates) 

    opt_updates, opt_state = optimizer.update(
        updates, state.opt_state, state.cache
    )
    opt_hessian = jax.tree.map(
        lambda h, c, p, g: h + (1.0-hessian_momentum) * 0.5 * 0.5 * (p - c) * g / rho**2, 
        state.hessian,
        state.cache, 
        params,
        updates) 


    cache = pick_one(first_step, params, state.cache)
    updates = pick_one(last_step, opt_updates, adv_updates)
    hessian = pick_one(last_step, opt_hessian, adv_hessian)

    adv_state = pick_one(last_step, state.adv_state, adv_state)
    opt_state = pick_one(last_step, opt_state, state.opt_state)

    steps_since_sync = (state.steps_since_sync + 1) % 2
    return updates, iFloodingState(
        steps_since_sync=steps_since_sync,
        opt_state=opt_state,
        adv_state=adv_state,
        cache=cache,
        hessian=hessian
    )

  return optax.GradientTransformation(init_fn, update_fn)









class SgdNoisyState(NamedTuple):
    ess: float    # effective sample size (lambda in the IVON paper)
    beta1: float
    weight_decay: float
    momentum: optax.Updates
    current_step: int = 0
    grad_acc: optax.Updates | None = None
    acc_count: int = 0


def sgd_noisy(
        learning_rate: optax.ScalarOrSchedule,
        ess: float,
        beta1: float = 0.9,
        weight_decay: float | None = 1e-4,
        clip_radius: float = float("inf")
) -> optax.GradientTransformation:
    
    weight_decay = 0. if weight_decay is None else weight_decay

    def sgd_noisy_init(params: optax.Params) -> SgdNoisyState:
        momentum = jax.tree.map(jnp.zeros_like, params)
        grad_acc = jax.tree.map(jnp.zeros_like, params)
        current_step = 1

        state = SgdNoisyState(
            ess, beta1, weight_decay, momentum,
            current_step, grad_acc, 0
        )
        return state

    def sgd_noisy_update(
        updates: optax.Updates,
        state: SgdNoisyState, #optax.OptState
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, SgdNoisyState]:
        (
            ess, beta1, weight_decay, momentum,
            current_step, grad_acc, acc_count
        ) = state

        # computing the averages of the accumulated gradients
        avg_grad = jax.tree.map(lambda g: g / acc_count, grad_acc)

        # computing momentum
        momentum = jax.tree.map(
            lambda m, g: beta1 * m + (1.0 - beta1) * g, 
            momentum,
            avg_grad)

        # debias term
        debias = 1.0 - beta1 ** current_step

        # add weight decay term to the gradient
        updates = jax.tree.map(
            lambda p, m: (m / debias + weight_decay * p) / (1. + weight_decay),
            params,
            momentum,
        )

        grad_acc = jax.tree.map(jnp.zeros_like, params)
        return updates, SgdNoisyState(
            ess, beta1, weight_decay, momentum, 
            current_step + 1, grad_acc, 0
        )

    sgd_noisy_optimizer = optax.GradientTransformation(sgd_noisy_init, sgd_noisy_update)

    # add gradient clipping and learning rate
    if clip_radius < float("inf"):
        sgd_noisy_optimizer = optax.chain(
            sgd_noisy_optimizer,
            optax.clip(clip_radius),
            optax.scale_by_learning_rate(learning_rate),
        )
    else:
        sgd_noisy_optimizer = optax.chain(
            sgd_noisy_optimizer, 
            optax.scale_by_learning_rate(learning_rate),
    )

    return sgd_noisy_optimizer



def sgd_noisy_accumulate_gradients(
        grads: optax.Updates,
        state: SgdNoisyState, #optax.OptState
) -> SgdNoisyState: #optax.OptState:
    (
        ess, beta1, weight_decay, momentum, 
        current_step, grad_acc, acc_count
    ) = state[0]

    grad_acc = jax.tree.map(lambda a, g: a + g, grad_acc, grads)

    state = (SgdNoisyState(
        ess, beta1, weight_decay, momentum, 
        current_step, grad_acc, acc_count + 1
    ), *state[1:])
    return state


def sgd_noisy_sample_parameters(
        rng: jax.random.PRNGKey,
        params: optax.Params,
        state: SgdNoisyState, #optax.OptState,
) -> tuple[optax.Params, SgdNoisyState]:
    (
        ess, beta1, weight_decay, momentum, 
        current_step, grad_acc, acc_count
    ) = state[0]

    # sample N(0,1) gaussian noise in tree shape
    tleaves, tdef = jax.tree.flatten(params)
    keys = jax.random.split(rng, len(tleaves))
    samples = [
        jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, tleaves)
    ]
    noise = jax.tree.unflatten(tdef, samples)

    # rescale noise
    noise = jax.tree.map(
        lambda n: n * jax.lax.rsqrt(ess * (1. + weight_decay)),
        noise
    )
    psample = jax.tree.map(lambda p, n: p + n, params, noise)

    state = (SgdNoisyState(
        ess, beta1, weight_decay, momentum, 
        current_step, grad_acc, acc_count
    ), *state[1:])
    return psample, state