import numpy as onp
from jax import numpy as jnp, random, tree_util as tree


def expand(struct, n, axis=0):
    return tree.tree_map(
        lambda x: jnp.broadcast_to(
            jnp.expand_dims(x, axis), x.shape[:axis] + (n,) + x.shape[axis:]
        ),
        struct,
    )


def shuffle_along_axis(rng, a, axis):
    idx = onp.array(random.uniform(rng, a.shape).argsort(axis=axis))
    return onp.take_along_axis(a, idx, axis=axis)


def split_rng_or_none(rng, n):
    if rng is None:
        # return [None] * n
        return jnp.zeros((n, 1), dtype=jnp.int32)
    else:
        return random.split(rng, n)


def use_self_as_default(*top_args, **are_functions):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            args = list(args)
            for arg in top_args:
                if isinstance(arg, tuple):
                    k0 = arg[0]
                    k1 = arg[1]
                else:
                    k0 = k1 = arg

                k_index = func.__code__.co_varnames.index(k0)
                if ((k0 not in kwargs) or (kwargs.get(k0) is None)) and (
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