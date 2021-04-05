import numpy as onp
import jax
from jax import numpy as jnp, random, tree_util as tree, pmap, partial


def _resize_batch_dim(array, num_devices=jax.local_device_count()):
    bsz = array.shape[0]
    assert (
        bsz % num_devices
    ) == 0, f"Batch size must be divisible but number of available devices, received batch size: {bsz} and num devices: {num_devices}"
    return array.reshape(num_devices, bsz // num_devices, *array.shape[1:])


def resize_batch_dim(struct, num_devices=jax.local_device_count()):
    return jax.tree_map(_resize_batch_dim, struct)


def flatten_dims(struct, dims=(1, 3)):
    return jax.tree_map(
        lambda t: t.reshape(
            *t.shape[: dims[0]],
            onp.prod(t.shape[dims[0] : dims[1]]),
            *t.shape[dims[1] :],
        ),
        struct,
    )


def is_sorted(a):
    return onp.all(a[:-1] <= a[1:])


def get_sharded_array_first(struct):
    return tree.tree_map(lambda x: x[0], struct)


def pmap_init(model, static_args, static_kwargs, *args, **kwargs):
    return pmap(partial(model.init, *static_args, **static_kwargs), axis_name="i")(
        *args, **kwargs
    )


def mean_of_f(f, mean_f=jnp.mean):
    def _f(*args, **kwargs):
        out = f(*args, **kwargs)
        if type(out) is tuple:
            return jnp.mean(out[0]), out[1:]
        return mean_f(out)

    return _f


def tree_flatten_array(struct, n=2):
    return tree.tree_map(
        lambda t: t.reshape(onp.prod(t.shape[:n]), *t.shape[n:]), struct
    )


def first_leaf_shape(struct):
    return tree.tree_flatten(struct)[0][0].shape


def tree_shape(struct):
    return tree.tree_map(jnp.shape, struct)


def expand(struct, n=1, axis=0):
    return tree.tree_map(
        lambda x: jnp.broadcast_to(
            jnp.expand_dims(x, axis), x.shape[:axis] + (n,) + x.shape[axis:]
        ),
        struct,
    )


def shuffle_along_axis(rng, a, axis):
    idx = onp.array(random.uniform(rng, a.shape).argsort(axis=axis))
    return onp.take_along_axis(a, idx, axis=axis)


def split_rng_or_none(rng, n=2):
    if rng is None:
        return [None] * n
    else:
        return random.split(rng, n)


def use_self_as_default(*top_args, **are_functions):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            args = list(args)

            # print(func)
            # print(len(args))
            # print(kwargs.keys())
            # print(kwargs)

            for arg in top_args:
                if isinstance(arg, tuple):
                    k0 = arg[0]
                    k1 = arg[1]
                else:
                    k0 = k1 = arg

                k_index = func.__code__.co_varnames.index(k0)
                # print(k0, k_index)

                if kwargs.get(k0) is not None:
                    continue
                elif ((k0 not in kwargs) or (kwargs.get(k0) is None)) and (
                    (len(args) < (k_index))
                ):
                    kwargs[k0] = getattr(self, k1)
                elif args[k_index - 1] is None:
                    args[k_index - 1] = getattr(self, k1)

            for func_name, func_kwargs in are_functions.items():
                f_index = func.__code__.co_varnames.index(func_name)
                not_in_kwargs = (func_name not in kwargs) or (
                    kwargs.get(func_name) is None
                )
                args_too_few = len(args) < (f_index)
                arg_is_none = args_too_few or (args[f_index - 1] is None)
                if (not_in_kwargs) and (args_too_few or arg_is_none):
                    _kwargs = {}
                    for fk in func_kwargs:
                        if fk in kwargs:
                            _kwargs[fk] = kwargs[fk]
                        elif len(args) > func.__code__.co_varnames.index(fk):
                            _kwargs[fk] = args[func.__code__.co_varnames.index(fk)]

                    if len(_kwargs) == len(func_kwargs):
                        if not_in_kwargs and args_too_few:
                            kwargs[func_name] = getattr(self, func_name)(**_kwargs)
                        elif arg_is_none:
                            args[f_index - 1] = getattr(self, func_name)(**_kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def call_self_as_default(top_kwargs):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for func_name, func_kwargs in top_kwargs.items():
                if (func_name not in kwargs) or (kwargs.get(func_name) is None):
                    _kwargs = {}
                    for fk in func_kwargs:
                        if fk in kwargs:
                            _kwargs[fk] = kwargs[fk]
                        elif len(args) > func.__code__.co_varnames.index(fk):
                            _kwargs[fk] = args[func.__code__.co_varnames.index(fk)]

                    if len(_kwargs) == len(func_kwargs):
                        kwargs[func_name] = getattr(self, func_name)(**_kwargs)

            return func(self, *args, **kwargs)

        return wrapper

    return decorator