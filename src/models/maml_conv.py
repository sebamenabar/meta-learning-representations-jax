import jax
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
            )(x, is_training=False, test_local_stats=True,)
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
        name=None,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.max_pool = max_pool
        self.layers = layers
        self.max_pool_factor = max_pool_factor
        self.activation = activation
        self.normalize = normalize

    def __call__(self, x, is_training):
        for _ in range(self.layers):
            x = ConvBlock(
                output_channels=self.output_channels,
                kernel_size=(3, 3),
                max_pool=self.max_pool,
                max_pool_factor=self.max_pool_factor,
                activation=self.activation,
                normalize=self.normalize,
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
    ):
        super().__init__(name=name)
        self.layers = layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.normalize = normalize
        self.max_pool = max_pool

    def __call__(self, x, is_training):
        x = ConvBase(
            output_channels=self.hidden_size,
            max_pool=self.max_pool,
            layers=self.layers,
            max_pool_factor=4 // self.layers,
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
    loss_fn, output_size, hidden_size, spatial_dims, max_pool, activation="relu"
):
    MiniImagenetCNNBody_t = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNBody(
            hidden_size=hidden_size, max_pool=max_pool, activation=activation,
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

