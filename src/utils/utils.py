import numpy as onp
from jax import numpy as jnp, random


def shuffle_along_axis(rng, a, axis):
    idx = onp.array(random.uniform(rng, a.shape).argsort(axis=axis))
    return onp.take_along_axis(a, idx, axis=axis)


def split_rng_or_none(rng, n):
    if rng is None:
        return jnp.empty((n, 0), dtype=jnp.int32)
    else:
        return random.split(rng, n)


def use_self_as_default(*top_args):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for arg in top_args:
                if isinstance(arg, tuple):
                    k0 = arg[0]
                    k1 = arg[1]
                else:
                    k0 = k1 = arg


                print(arg, func.__code__.co_varnames.index(k0))

                if ((k0 not in kwargs) or (kwargs.get(k0) is None)) and (
                    len(*args) < (func.__code__.co_varnames.index(k0) - 1)
                ):
                    kwargs[k0] = getattr(self, k1)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def call_self_as_default(top_kwargs):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for func_name, func_kwargs in top_kwargs.items():
                if (func_name not in kwargs) or (kwargs.get(func_name) is None):
                    kwargs[func_name] = getattr(self, func_name)(
                        **{fk: kwargs[fk] for fk in func_kwargs}
                    )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator