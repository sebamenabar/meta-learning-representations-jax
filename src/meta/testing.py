import jax
from jax import numpy as jnp
from utils import expand
from .wrappers import ContinualLearnerB


class ContinualTesterB(ContinualLearnerB):
    def __init__(self, *args, preprocess_fn=None, **kwargs):
        self.preprocess_fn = preprocess_fn
        super().__init__(*args, **kwargs)

    @jax.partial(jax.jit, static_argnums=0)
    def jit_inner_loop(self, x, y, params, state, fast_params, fast_state, lr):
        x = x / 255
        if self.preprocess_fn:
            x = self.preprocess_fn(x)
        return self.inner_loop(
            x,
            y,
            params=params,
            state=state,
            fast_params=fast_params,
            fast_state=fast_state,
            training=False,
            lr=lr,
        )

    def _adapt_to_loader(self, loader, params, state, lr):
        fast_params = expand(self.get_fp(params))
        fast_state = expand(self.get_fs(state))

        for x_spt, y_spt in loader:
            # print(tree_shape((x_spt, y_spt)))
            out = self.jit_inner_loop(
                expand(x_spt),
                expand(y_spt),
                params=params,
                state=state,
                fast_params=fast_params,
                fast_state=fast_state,
                lr=lr,
            )
            fast_params = out["fast_params"]
            fast_state = out["fast_state"]

        return out

    @jax.partial(jax.jit, static_argnums=0)
    def jit_forward(self, x, y, params, state, fast_params, fast_state):
        x = x / 255
        if self.preprocess_fn:
            x = self.preprocess_fn(x)
        out = self.apply_and_loss(
            x,
            y,
            params=params,
            state=state,
            fast_params=fast_params,
            fast_state=fast_state,
            training=False,
        )
        return out["loss_aux"]["acc"], out["loss_aux"]["loss"]

    def get_loader_accuracy(self, loader, params, state, fast_params, fast_state):
        accs = []
        losses = []
        for x, y in loader:
            x = expand(x)
            y = expand(y)
            acc, loss = self.jit_forward(x, y, params, state, fast_params, fast_state)
            accs.append(acc.reshape(-1))
            losses.append(loss.reshape(-1))

        accs = jnp.concatenate(accs)
        losses = jnp.concatenate(losses)

        return accs, losses

    def test(self, train_loader, test_loader, params, state, lr, mean=True):
        adaptation_out = self._adapt_to_loader(train_loader, params, state, lr)
        fast_params, fast_state = (
            adaptation_out["fast_params"],
            adaptation_out["fast_state"],
        )
        train_accs, train_losses = self.get_loader_accuracy(
            train_loader, params, state, fast_params, fast_state
        )
        if mean:
            train_accs = train_accs.mean()
            train_losses = train_losses.mean()
        if test_loader is not None:
            test_accs, test_losses = self.get_loader_accuracy(
                test_loader, params, state, fast_params, fast_state
            )
            if mean:
                test_accs = test_accs.mean()
                test_losses = test_losses.mean()
        else:
            test_accs = test_losses = None

        return train_accs, train_losses, test_accs, test_losses