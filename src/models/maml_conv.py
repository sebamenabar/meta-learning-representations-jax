from argparse import ArgumentParser

import jax
import jax.numpy as jnp
from jax.random import split
import haiku as hk

from .layers import build_initializer, MyBatchNorm, Affine
from .activations import activations
from typing import Optional, Sequence


def miniimagenet_cnn_argparse(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--hidden_size", default=32, type=int)
    # parser.add_argument("--track_bn_stats", default=False, action="store_true")
    parser.add_argument(
        "--activation", type=str, default="relu", choices=list(activations.keys())
    )
    return parser


class CustomNorm(hk.Module):
    """Normalizes inputs to maintain a mean of ~0 and stddev of ~1.
    See: https://arxiv.org/abs/1502.03167.
    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:
      - No scale/offset in which case ``create_*`` should be set to ``False`` and
        ``scale``/``offset`` aren't passed when the module is called.
      - Trainable scale/offset in which case ``create_*`` should be set to
        ``True`` and again ``scale``/``offset`` aren't passed when the module is
        called. In this case this module creates and owns the ``scale``/``offset``
        variables.
      - Externally generated ``scale``/``offset``, such as for conditional
        normalization, in which case ``create_*`` should be set to ``False`` and
        then the values fed in at call time.
    NOTE: ``jax.vmap(hk.transform(BatchNorm))`` will update summary statistics and
    normalize values on a per-batch basis; we currently do *not* support
    normalizing across a batch axis introduced by vmap.
    """

    def __init__(
        self,
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        use_stats_during_training: bool = True,
        eps: float = 1e-5,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        cross_replica_axis: Optional[str] = None,
        data_format: str = "channels_last",
        name: Optional[str] = None,
    ):
        """Constructs a BatchNorm module.
        Args:
          create_scale: Whether to include a trainable scaling factor.
          create_offset: Whether to include a trainable offset.
          decay_rate: Decay rate for EMA.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). Can only be set
            if ``create_scale=True``. By default, ``1``.
          offset_init: Optional initializer for bias (aka offset). Can only be set
            if ``create_offset=True``. By default, ``0``.
          axis: Which axes to reduce over. The default (``None``) signifies that all
            but the channel axis should be normalized. Otherwise this is a list of
            axis indices which will have normalization statistics calculated.
          cross_replica_axis: If not ``None``, it should be a string representing
            the axis name over which this module is being run within a ``jax.pmap``.
            Supplying this argument means that batch statistics are calculated
            across all replicas on that axis.
          data_format: The data format of the input. Can be either
            ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
            default it is ``channels_last``.
          name: The module name.
        """
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")

        self.use_stats_during_training = use_stats_during_training
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.cross_replica_axis = cross_replica_axis
        if data_format == "channels_last":
            self.channel_index = -1
        # self.channel_index = hk._src.get_channel_index(data_format)
        self.mean_ema = hk.ExponentialMovingAverage(
            decay_rate, name="mean_ema", zero_debias=False, warmup_length=100
        )
        self.var_ema = hk.ExponentialMovingAverage(
            decay_rate, name="var_ema", zero_debias=False, warmup_length=100
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes the normalized version of the input.
        Args:
          inputs: An array, where the data format is ``[..., C]``.
          is_training: Whether this is during training.
          test_local_stats: Whether local stats are used when is_training=False.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.
        Returns:
          The array, normalized across all but the last dimension.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        if is_training or test_local_stats:
            cross_replica_axis = self.cross_replica_axis
            if self.cross_replica_axis:
                mean = jnp.mean(inputs, axis, keepdims=True)
                mean = jax.lax.pmean(mean, cross_replica_axis)
                mean_of_squares = jnp.mean(inputs ** 2, axis, keepdims=True)
                mean_of_squares = jax.lax.pmean(mean_of_squares, cross_replica_axis)
                var = mean_of_squares - mean ** 2
            else:
                mean = jnp.mean(inputs, axis, keepdims=True)
                # This uses E[(X - E[X])^2].
                # TODO(tycai): Consider the faster, but possibly less stable
                # E[X^2] - E[X]^2 method.
                var = jnp.var(inputs, axis, keepdims=True)

        if is_training:
            self.mean_ema(mean)
            self.var_ema(var)

        if self.use_stats_during_training or not is_training:
            mean = self.mean_ema.average
            var = self.var_ema.average

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
        elif scale is None:
            scale = jnp.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = jnp.zeros([], dtype=w_dtype)

        inv = scale * jax.lax.rsqrt(var + self.eps)
        return (inputs - mean) * inv + offset


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
        self.initializer = build_initializer(
            activation, name=initializer, truncated=False
        )
        # if initializer == "glorot_uniform":
        #     self.initializer = hk.initializers.VarianceScaling(
        #         1.0, "fan_avg", "uniform"
        #     )
        # elif initializer == "kaiming_normal":
        #     self.initializer = hk.initializers.VarianceScaling(
        #         2.0, "fan_out", "truncated_normal"
        #     )
        # else:
        #     raise NameError(f"Unknown initializer {initializer}")

    def __call__(self, x, is_training):
        x = hk.Conv2D(
            output_channels=self.output_channels,
            kernel_shape=self.kernel_size,
            stride=self.conv_stride,
            padding="SAME",
            w_init=self.initializer,
            # b_init=hk.initializers.Constant(0.0),
        )(x)
        if not self.norm_before_act:
            x = self.activation(x)

        if self.normalize == "bn":
            print("batch norm")
            x = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.9
                if self.track_stats
                else 0.0,  # 0 for no tracking of stats
            )(x, is_training=is_training, test_local_stats=not self.track_stats,)

        elif self.normalize == "gn":
            print("group norm")
            x = hk.GroupNorm(groups=4)(x)
        elif self.normalize == "in":
            print("in norm")
            x = hk.InstanceNorm(create_scale=True, create_offset=True)(x)
        elif self.normalize == "ln":
            print("ln norm")
            x = hk.LayerNorm(
                axis=slice(1, None, None), create_scale=True, create_offset=True
            )(x)
        elif self.normalize == "custom":
            print("custom norm")
            x = MyBatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.9,
                # use_stats_during_training=False,
                #  axis=slice(1, None, None),
            )(x, is_training)
        elif self.normalize == "affine":
            print("Affine norm")
            x = Affine()(x)
        else:
            print("No norm")

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
                decay_rate=0.9
                if self.track_stats
                else 0.0,  # 0 for no tracking of stats
            )(x, is_training=is_training, test_local_stats=not self.track_stats,)
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
