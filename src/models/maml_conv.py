from argparse import ArgumentParser

import jax
import jax.numpy as jnp
from jax.random import split
import haiku as hk
from .activations import activations


def miniimagenet_cnn_argparse(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--hidden_size", default=32, type=int)
    # parser.add_argument("--track_bn_stats", default=False, action="store_true")
    parser.add_argument(
        "--activation", type=str, default="relu", choices=list(activations.keys())
    )
    return parser


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
        initializer="glorot_uniform",
        norm_before_act=True,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self.max_pool_factor = max_pool_factor
        self.normalize = normalize
        self.stride = stride = (int(2 * max_pool_factor), int(2 * max_pool_factor), 1)
        self.norm_before_act = norm_before_act
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
        if initializer == "glorot_uniform":
            self.initializer = hk.initializers.VarianceScaling(
                1.0, "fan_avg", "uniform"
            )
        elif initializer == "kaiming_normal":
            self.initializer = hk.initializers.VarianceScaling(
                2.0, "fan_out", "truncated_normal"
            )
        else:
            raise NameError(f"Unknown initializer {initializer}")

    def __call__(self, x, is_training):
        x = hk.Conv2D(
            output_channels=self.output_channels,
            kernel_shape=self.kernel_size,
            stride=self.conv_stride,
            padding="SAME",
            w_init=self.initializer,
            b_init=hk.initializers.Constant(0.0),
        )(x)
        if not self.norm_before_act:
            x = self.activation(x)
        if self.normalize == "bn":
            x = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.95
                if self.track_stats
                else 0.0,  # 0 for no tracking of stats
            )(
                x,
                is_training=self.track_stats and is_training,
                test_local_stats=not self.track_stats,
            )

        elif self.normalize == "gn":
            x = hk.GroupNorm(groups=4)(x)
        elif self.normalize == "in":
            x = hk.InstanceNorm(create_scale=True, create_offset=True)(x)
        elif self.normalize == "ln":
            x = hk.LayerNorm(
                axis=slice(1, None, None), create_scale=True, create_offset=True
            )(x)

        if self.norm_before_act:
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
        initializer=None,
        norm_before_act=True,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.max_pool = max_pool
        self.layers = layers
        self.max_pool_factor = max_pool_factor
        self.activation = activation
        self.normalize = normalize
        self.track_stats = track_stats
        self.initializer = initializer
        self.norm_before_act = norm_before_act

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
                initializer=self.initializer,
                norm_before_act=self.norm_before_act,
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
        initializer=None,
        norm_before_act=True,
        final_norm=False,
        avg_pool=False,
    ):
        super().__init__(name=name)
        self.layers = layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.normalize = normalize
        self.max_pool = max_pool
        self.track_stats = track_stats
        self.initializer = initializer
        self.norm_before_act = norm_before_act
        self.final_norm = final_norm
        self.avg_pool = avg_pool

    def __call__(self, x, is_training):
        x = ConvBase(
            output_channels=self.hidden_size,
            max_pool=self.max_pool,
            layers=self.layers,
            max_pool_factor=4 // self.layers,
            track_stats=self.track_stats,
            initializer=self.initializer,
            norm_before_act=self.norm_before_act,
            normalize=self.normalize,
        )(x, is_training)
        if self.avg_pool:
            x = hk.avg_pool(x, (1, 5, 5, 1), 1, padding="VALID", channel_axis=3)
            x = hk.Reshape((self.hidden_size,))(x)
        if self.final_norm == "bn":
            x = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.95
                if self.track_stats
                else 0.0,  # 0 for no tracking of stats
            )(
                x,
                is_training=self.track_stats and is_training,
                test_local_stats=not self.track_stats,
            )
        elif self.final_norm == "gn":
            x = hk.GroupNorm(4)(x)
        elif self.final_norm == "in":
            x = hk.InstanceNorm(create_scale=True, create_offset=True)(x)
        return (x,)


class MiniImagenetCNNHead(hk.Module):
    def __init__(
        self,
        output_size,
        spatial_dims=25,
        hidden_size=32,
        name=None,
        avg_pool=False,
        initializer="glorot_uniform",
        head_bias=True,
    ):
        super().__init__(name=name)
        self.head_bias = head_bias
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.spatial_dims = spatial_dims
        self.avg_pool = avg_pool
        if initializer == "glorot_uniform":
            self.initializer = hk.initializers.VarianceScaling(
                1.0, "fan_avg", "uniform"
            )
        elif initializer == "kaiming_normal":
            self.initializer = hk.initializers.VarianceScaling(
                2.0, "fan_out", "truncated_normal"
            )

    def __call__(self, x, is_training):
        if self.avg_pool:
            pass
        else:
            x = hk.Reshape((self.spatial_dims * self.hidden_size,))(x)
        x = hk.Linear(
            self.output_size,
            with_bias=self.head_bias,
            w_init=self.initializer,
            b_init=hk.initializers.Constant(0.0),
        )(x)
        return x


def make_miniimagenet_cnn(
    output_size,
    hidden_size,
    spatial_dims,
    max_pool,
    initializer,
    avg_pool=False,
    activation="relu",
    track_stats=False,
    head_bias=True,
    norm_before_act=True,
    final_norm=False,
    normalize=True,
):
    MiniImagenetCNNBody_t = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNBody(
            hidden_size=hidden_size,
            max_pool=max_pool,
            activation=activation,
            track_stats=track_stats,
            initializer=initializer,
            norm_before_act=norm_before_act,
            final_norm=final_norm,
            normalize=normalize,
            avg_pool=avg_pool,
        )(
            x, is_training,
        )
    )
    MiniImagenetCNNHead_t = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNHead(
            output_size=output_size,
            hidden_size=hidden_size,
            spatial_dims=spatial_dims,
            initializer=initializer,
            avg_pool=avg_pool,
            head_bias=head_bias,
        )(
            x, is_training,
        )
    )

    return (
        MiniImagenetCNNBody_t,
        MiniImagenetCNNHead_t,
    )


def make_params(rng, dataset, slow_init, slow_apply, fast_init):
    slow_rng, fast_rng = split(rng)
    if dataset == "miniimagenet":
        setup_tensor = jnp.zeros((2, 84, 84, 3))
    elif dataset == "omniglot":
        setup_tensor = jnp.zeros((2, 28, 28, 1))
    slow_params, slow_state = slow_init(slow_rng, setup_tensor, True)
    slow_outputs, _ = slow_apply(slow_params, slow_state, slow_rng, setup_tensor, True)
    fast_params, fast_state = fast_init(fast_rng, *slow_outputs, True)

    return slow_params, fast_params, slow_state, fast_state


def prepare_model(
    dataset,
    output_size,
    hidden_size,
    activation,
    initializer,
    avg_pool=False,
    track_stats=False,
    head_bias=True,
    norm_before_act=True,
    final_norm=False,
    normalize=True,
):
    if dataset == "miniimagenet":
        max_pool = True
        spatial_dims = 25
    elif dataset == "omniglot":
        max_pool = False
        spatial_dims = 4

    return make_miniimagenet_cnn(
        output_size=output_size,
        hidden_size=hidden_size,
        spatial_dims=spatial_dims,
        max_pool=max_pool,
        activation=activation,
        track_stats=track_stats,
        initializer=initializer,
        avg_pool=avg_pool,
        head_bias=head_bias,
        norm_before_act=norm_before_act,
        final_norm=final_norm,
        normalize=normalize,
    )

