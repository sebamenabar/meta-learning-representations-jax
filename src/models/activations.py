import jax

activations = [
    "celu",
    "elu",
    "gelu",
    "glu",
    "leaky_relu",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "soft_sign",
    "softplus",
    # "silu",
    "swish",
]


activations = {name: getattr(jax.nn, name) for name in activations}


def mish(x):
    return x * jax.numpy.tanh(jax.nn.softplus(x))


activations["mish"] = mish
