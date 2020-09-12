import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk


def build_initializer(nonlinearity, name, truncated=False):
    if name == "glorot_uniform":
        return hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
    elif name == "kaiming_normal":
        if nonlinearity == "relu":
            scale = 2
        elif nonlinearity == "leaky_relu":
            scale = 2 / (1 + 0.01)
        return hk.initializers.VarianceScaling(
            scale, "fan_out", "truncated_normal" if truncated else "normal"
        )  # rts uses fan out and pytorch uses normal distribution instead of truncated normal


class MyExponentialMovingAverage(hk.Module):
    """Maintains an exponential moving average.
    This uses the Adam debiasing procedure.
    See https://arxiv.org/pdf/1412.6980.pdf for details.
    """

    def __init__(
        self, decay, zero_debias=True, warmup_length=0, init=jnp.zeros, name=None
    ):
        """Initializes an ExponentialMovingAverage module.
        Args:
          decay: The chosen decay. Must in [0, 1). Values close to 1 result in slow
            decay; values close to 0 result in fast decay.
          zero_debias: Whether to run with zero-debiasing.
          warmup_length: A positive integer, EMA has no effect until
            the internal counter has reached `warmup_length` at which point the
            initial value for the decaying average is initialized to the input value
            after `warmup_length` iterations.
          name: The name of the module.
        """
        super().__init__(name=name)
        self.init = init
        self._decay = decay
        if warmup_length < 0:
            raise ValueError(
                f"`warmup_length` is {warmup_length}, but should be non-negative."
            )
        self._warmup_length = warmup_length
        self._zero_debias = zero_debias
        if warmup_length and zero_debias:
            raise ValueError(
                "Zero debiasing does not make sense when warming up the value of the "
                "average to an initial value. Set zero_debias=False if setting "
                "warmup_length to a non-zero value."
            )

    def initialize(self, shape, dtype=jnp.float32):
        """If uninitialized sets the average to ``zeros`` of the given shape/dtype."""
        if hasattr(shape, "shape"):
            warnings.warn(
                "Passing a value into initialize instead of a shape/dtype "
                "is deprecated. Update your code to use: "
                "`ema.initialize(v.shape, v.dtype)`.",
                category=DeprecationWarning,
            )
            shape, dtype = shape.shape, shape.dtype

        hk.get_state("hidden", shape, dtype, init=self.init)
        hk.get_state("average", shape, dtype, init=self.init)

    def __call__(self, value, update_stats=True):
        """Updates the EMA and returns the new value.
        Args:
          value: The array-like object for which you would like to perform an
            exponential decay on.
          update_stats: A Boolean, whether to update the internal state
            of this object to reflect the input value. When `update_stats` is False
            the internal stats will remain unchanged.
        Returns:
          The exponentially weighted average of the input value.
        """
        if not isinstance(value, jnp.ndarray):
            value = jnp.asarray(value)

        counter = hk.get_state(
            "counter",
            (),
            jnp.int32,
            init=hk.initializers.Constant(-self._warmup_length),
        )
        counter = counter + 1

        decay = jax.lax.convert_element_type(self._decay, value.dtype)
        if self._warmup_length > 0:
            decay = jax.lax.select(counter <= 0, 0.0, decay)

        one = jnp.ones([], value.dtype)
        hidden = hk.get_state("hidden", value.shape, value.dtype, init=self.init)
        hidden = hidden * decay + value * (one - decay)

        average = hidden
        if self._zero_debias:
            average /= one - jnp.power(decay, counter)

        if update_stats:
            hk.set_state("counter", counter)
            hk.set_state("hidden", hidden)
            hk.set_state("average", average)

        return average

    @property
    def average(self):
        return hk.get_state("average")


class MyBatchNorm(hk.Module):
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

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.cross_replica_axis = cross_replica_axis
        # self.channel_index = utils.get_channel_index(data_format)
        if data_format == "channels_last":
            self.channel_index = -1
        self.mean_ema = MyExponentialMovingAverage(
            decay_rate, zero_debias=False, init=jnp.zeros, name="mean_ema"
        )
        self.var_ema = MyExponentialMovingAverage(
            decay_rate, zero_debias=False, init=jnp.ones, name="var_ema"
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
        else:
            mean = self.mean_ema.average
            var = self.var_ema.average

        if is_training:
            self.mean_ema(mean)
            self.var_ema(var)

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        inv = scale * jax.lax.rsqrt(var + self.eps)
        return (inputs - mean) * inv + offset
