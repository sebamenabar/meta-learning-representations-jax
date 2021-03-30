import jax
import haiku as hk


def filter_structure_by_str(instr, structure):
    return hk.data_structures.filter(
        lambda module_name, name, value: instr in module_name, structure
    )


class SimpleModel(hk.Module):
    train_slow_phase = "train_slow_phase"
    train_fast_phase = "train_fast_phase"
    test_slow_phase = "test_slow_phase"
    test_fast_phase = "test_fast_phase"

    get_train_slow_params = jax.partial(filter_structure_by_str, "slow")
    get_train_fast_params = jax.partial(filter_structure_by_str, "fast")
    get_train_slow_state = jax.partial(filter_structure_by_str, "slow")
    get_train_fast_state = jax.partial(filter_structure_by_str, "fast")

    get_test_slow_params = jax.partial(filter_structure_by_str, "slow")
    get_test_fast_params = jax.partial(filter_structure_by_str, "fast")
    get_test_slow_state = jax.partial(filter_structure_by_str, "slow")
    get_test_fast_state = jax.partial(filter_structure_by_str, "fast")

    def __call__(self, inputs, phase, training):
        if phase in [self.train_slow_phase, self.test_slow_phase, "all"]:
            with hk.experimental.name_scope("slow1"):
                x = inputs
                x = hk.Linear(8)(x)
                x = hk.BatchNorm(True, True, 0.99)(x, training)
                out = (x,)
        else:
            (x,) = inputs

        if phase in [self.train_fast_phase, self.test_fast_phase, "all"]:

            with hk.experimental.name_scope("fast"):
                x = hk.Linear(8)(x)
                x = hk.BatchNorm(True, True, 0.99)(x, training)

            with hk.experimental.name_scope("slow2"):
                x = hk.Linear(8)(x)
                x = hk.BatchNorm(True, True, 0.99)(x, training)

            with hk.experimental.name_scope("fast"):
                x = hk.Linear(4)(x)
                x = hk.BatchNorm(True, True, 0.99)(x, training)
                x = hk.Linear(2, w_init=hk.initializers.Constant(0))(x)

            out = x

        return out