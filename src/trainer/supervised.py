import jax
from jax import value_and_grad
from jax.random import split

import optax as ox

class ModelWrapper:
    def __init__(
        self,
        forward,
        params,
        state,
        opt_state=None,
    ):
        self.__forward = forward
        self.params = params
        self.state = state
        self.opt_state = opt_state

        self.__step_fn = lambda x: x

    def __call__(self, inputs, rng=None, training=None):
        return self.__forward(self.params, self.state, rng, inputs, training)
    
    @jax.partial(jax.jit, static_argnums=0)
    def _jit_call_train(self, params, state, rng, inputs, training):
        return self.__forward(params, state, rng, inputs, training)
        
    def jit_call_train(self, rng, inputs):
        return self.__forward(self.params, self.state, rng, inputs, True)
        
    @jax.partial(jax.jit, static_argnums=0)
    def _jit_call_validate(self, params, state, inputs):
        return self.__forward(params, state, None, inputs, False)
    
    def jit_call_validate(self, inputs):
        return self._jit_call_validate(self.params, self.state, inputs)
    
    def set_step_fn(self, fn):
        self.__step_fn = fn
        return self

    def init_opt_state(self, optimizer):
        self.opt_state = optimizer.init(self.params)
        return self

    def make_step_fn(self, optimizer, scheduler_fn, loss_fn, preprocess_fn):
        def __grad_fn(params, state, rng, inputs, targets):
            preds, state = self.__forward(
                params, state, rng, inputs, is_training=True
            )
            loss, aux = loss_fn(preds, targets)
            return loss, (state, aux)

        def __step_fn(step_num, rng, params, state, opt_state, inputs, targets):
            rng_pre, rng_step = split(rng)
            inputs = preprocess_fn(rng_pre, inputs)
            (loss, (state, aux)), grads = value_and_grad(__grad_fn, has_aux=True)(
                params,
                state,
                rng_step,
                inputs,
                targets,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            updates = scheduler_fn(step_num, updates)
            params = ox.apply_updates(params, updates)

            return params, state, opt_state, loss, aux

        return __step_fn

    def train_step(self, step_num, rng, inputs, targets, step_fn=None):
        if step_fn is None:
            step_fn = self.__step_fn

        (self.params, self.state, self.opt_state, loss, aux,) = step_fn(
            step_num,
            rng,
            self.params,
            self.state,
            self.opt_state,
            inputs,
            targets,
        )

        return loss, aux