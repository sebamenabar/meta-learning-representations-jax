import jax
import jax.numpy as jnp
import haiku as hk
from .activations import activations


class ConvBlock(hk.Module):
    def __init__(
        self,
        output_channels,
        kernel_size,
        max_pool=True,
        max_pool_factor=1.0,
        activation="relu",
        normalize=True,
        name=None,
        track_stats=False,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self.max_pool_factor = max_pool_factor
        self.normalize = normalize
        self.stride = stride = (int(2 * max_pool_factor), int(2 * max_pool_factor), 1)
        # if activation == "relu":
        #     self.activation = jax.nn.relu
        # elif activation == "swish":
        #     self.activation = jax.nn.swish
        self.activation = activations[activation]
        if max_pool:
            self.conv_stride = (1, 1)
        else:
            self.conv_stride = self.stride[:2]
        # self.conv_stride = (1, 1)
        self.track_stats = track_stats

    def __call__(self, x, is_training):
        x = hk.Conv2D(
            output_channels=self.output_channels,
            kernel_shape=self.kernel_size,
            stride=self.conv_stride,
            padding="SAME",
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            b_init=hk.initializers.Constant(0.0),
        )(x)
        if self.normalize:
            x = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.0,  # 0 for no tracking of stats
            )(
                x,
                is_training=self.track_stats and is_training,
                test_local_stats=not self.track_stats,
            )
        x = self.activation(x)
        if self.max_pool:
            x = hk.MaxPool(
                window_shape=self.stride, strides=self.stride, padding="VALID"
            )(x)
        return x


class ConvBase(hk.Module):
    def __init__(
        self,
        output_channels,
        max_pool=True,
        layers=4,
        max_pool_factor=1.0,
        activation="relu",
        normalize=True,
        track_stats=False,
        name=None,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.max_pool = max_pool
        self.layers = layers
        self.max_pool_factor = max_pool_factor
        self.activation = activation
        self.normalize = normalize
        self.track_stats = track_stats

    def __call__(self, x, is_training):
        for _ in range(self.layers):
            x = ConvBlock(
                output_channels=self.output_channels,
                kernel_size=(3, 3),
                max_pool=self.max_pool,
                max_pool_factor=self.max_pool_factor,
                activation=self.activation,
                normalize=self.normalize,
                track_stats=self.track_stats,
            )(x, is_training)
        return x


class MiniImagenetCNNBody(hk.Module):
    def __init__(
        self,
        hidden_size=32,
        layers=4,
        activation="relu",
        normalize=True,
        # name="mini_imagenet_cnn",
        name=None,
        max_pool=True,
        track_stats=False,
    ):
        super().__init__(name=name)
        self.layers = layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.normalize = normalize
        self.max_pool = max_pool
        self.track_stats = track_stats

    def __call__(self, x, is_training):
        x = ConvBase(
            output_channels=self.hidden_size,
            max_pool=self.max_pool,
            layers=self.layers,
            max_pool_factor=4 // self.layers,
            track_stats=self.track_stats,
        )(x, is_training)
        return (x,)


class MiniImagenetCNNHead(hk.Module):
    def __init__(
        self, output_size, spatial_dims=25, hidden_size=32, name=None,
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.spatial_dims = spatial_dims

    def __call__(self, x, is_training):
        x = hk.Reshape((self.spatial_dims * self.hidden_size,))(x)
        x = hk.Linear(
            self.output_size,
            with_bias=True,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            b_init=hk.initializers.Constant(0.0),
        )(x)
        return x


def MiniImagenetCNNMaker(
    loss_fn,
    output_size,
    hidden_size,
    spatial_dims,
    max_pool,
    activation="relu",
    track_stats=False,
):
    MiniImagenetCNNBody_t = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNBody(
            hidden_size=hidden_size,
            max_pool=max_pool,
            activation=activation,
            track_stats=track_stats,
        )(
            x, is_training,
        )
    )
    MiniImagenetCNNHead_t = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNHead(
            output_size=output_size, hidden_size=hidden_size, spatial_dims=spatial_dims
        )(
            x, is_training,
        )
    )

    def slow_apply(rng, slow_params, slow_state, is_training, inputs):
        return MiniImagenetCNNBody_t.apply(
            slow_params, slow_state, rng, inputs, is_training,
        )

    def fast_apply_and_loss_fn(
        rng, fast_params, fast_state, is_training, inputs, targets
    ):
        # params = hk.data_structures.merge(slow_params, fast_params)
        logits, state = MiniImagenetCNNHead_t.apply(
            fast_params, fast_state, rng, inputs, is_training
        )
        loss, *aux = loss_fn(logits, targets)
        return loss, (state, *aux)

        return logits, state

    return (
        MiniImagenetCNNBody_t,
        MiniImagenetCNNHead_t,
        slow_apply,
        fast_apply_and_loss_fn,
    )


def prepare_model(loss_fn, dataset, way, hidden_size, activation, track_stats=False):
    if dataset == "miniimagenet":
        max_pool = True
        spatial_dims = 25
    elif dataset == "omniglot":
        max_pool = False
        spatial_dims = 4

    return MiniImagenetCNNMaker(
        loss_fn,
        output_size=way,
        hidden_size=hidden_size,
        spatial_dims=spatial_dims,
        max_pool=max_pool,
        activation=activation,
        track_stats=track_stats,
    )


def make_params(rng, dataset, slow_init, slow_apply, fast_init, device):
    if dataset == "miniimagenet":
        setup_tensor = jnp.zeros((2, 84, 84, 3))
    elif dataset == "omniglot":
        setup_tensor = jnp.zeros((2, 28, 28, 1))
    slow_params, slow_state = slow_init(rng, setup_tensor, True)
    slow_outputs, _ = slow_apply(rng, slow_params, slow_state, True, setup_tensor,)
    fast_params, fast_state = fast_init(rng, *slow_outputs, True)
    move_to_device = lambda x: jax.device_put(x, device)
    slow_params = jax.tree_map(move_to_device, slow_params)
    fast_params = jax.tree_map(move_to_device, fast_params)
    slow_state = jax.tree_map(move_to_device, slow_state)
    fast_state = jax.tree_map(move_to_device, fast_state)

    return slow_params, fast_params, slow_state, fast_state
