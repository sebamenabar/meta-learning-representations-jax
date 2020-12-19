import jax
import haiku as hk
# from typing_extensions import final

from models.layers import Affine, build_initializer, get_norm


class OMLConvnet(hk.Module):
    def __init__(
        self,
        output_size,
        spatial_dims,
        num_fc_layers=2,
        num_adaptation_fc=2,
        fc_hidden_size=1024,
        avg_pool=False,
        head_bias=False,
        conv_hidden_size=256,
        # num_conv_blocks=6,
        normalization="affine",
        initializer="kaiming",
        name=None,
        final_pool="none",
        strides=(1, 1, 1, 2, 1, 2),
    ):
        super().__init__(name=name)
        self.hidden_size = conv_hidden_size
        #  self.num_layers = num_layers
        self.normalization = normalization
        self.initializer = initializer
        self.avg_pool = avg_pool
        self.spatial_dims = spatial_dims

        # Not break other stuff
        if avg_pool:
            final_pool = "avg"
        self.final_pool = final_pool

        self.blocks = []
        for stride in strides:
            self.blocks.append(
                OMLBlock(conv_hidden_size, normalization, initializer, stride)
            )

        self.fcs = []
        self.norms = []
        self.num_adaptation_fc = num_adaptation_fc
        self.normalization = normalization
        self.num_fc_layers = num_fc_layers
        normalization = get_norm(normalization)
        w_init = build_initializer("relu", initializer)
        for i in range(num_fc_layers - 1):
            self.fcs.append(
                hk.Linear(
                    fc_hidden_size,
                    w_init=w_init,
                    b_init=hk.initializers.Constant(0.0),
                    # name=f"encoder_linear_{i}",
                )
            )
            self.norms.append(normalization(f"encoder_norm_{i}"))

        self.classifier = hk.Linear(
            output_size,
            w_init=w_init,
            with_bias=head_bias,
            b_init=hk.initializers.Constant(0.0),
            name="classifier_linear",
        )

    def __call__(self, x, is_training, phase, num_adaptation_fc=None):
        assert phase in ["all", "encoder", "adaptation"]
        if num_adaptation_fc is None:
            num_adaptation_fc = self.num_adaptation_fc

        if phase in ["encoder", "all"]:
            for block in self.blocks:
                x = block(x, is_training)
            if self.final_pool == "avg":
                x = hk.avg_pool(
                    x,
                    jax.numpy.array([1, x.shape[1], x.shape[2], 1]),
                    1,
                    padding="VALID",
                    channel_axis=3,
                )
                x = hk.Reshape((self.hidden_size,))(x)
            elif self.final_pool == "max":
                x = hk.max_pool(
                    x,
                    jax.numpy.array([1, x.shape[1], x.shape[2], 1]),
                    1,
                    padding="VALID",
                    channel_axis=3,
                )
                x = hk.Reshape((self.hidden_size,))(x)
            else:
                x = hk.Reshape((self.spatial_dims * self.hidden_size,))(x)

            for fc, norm in zip(
                self.fcs[: self.num_fc_layers - num_adaptation_fc],
                self.norms[: self.num_fc_layers - num_adaptation_fc],
            ):
                x = fc(x)
                if self.normalization == "bn":
                    x = norm(x, is_training)
                else:
                    x = norm(x)
                x = jax.nn.relu(x)
            if phase == "encoder":
                x = (x,)

        if phase in ["adaptation", "all"]:
            for fc, norm in zip(
                self.fcs[self.num_fc_layers - num_adaptation_fc :],
                self.norms[self.num_fc_layers - num_adaptation_fc :],
            ):
                x = fc(x)
                if self.normalization == "bn":
                    x = norm(x, is_training)
                else:
                    x = norm(x)
                x = jax.nn.relu(x)

            x = self.classifier(x)

        return x


class OMLBlock(hk.Module):
    def __init__(
        self,
        hidden_size,
        normalization,
        initializer,
        stride,
        name=None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.normalization = normalization
        self.norm = get_norm(normalization)(None)
        self.initializer = initializer
        initializer = build_initializer("relu", initializer)
        self.stride = stride

        self.conv = hk.Conv2D(
            output_channels=hidden_size,
            kernel_shape=3,
            stride=stride,
            padding="VALID",
            w_init=initializer,
        )

    def __call__(self, x, is_training):
        x = self.conv(x)
        if self.normalization == "bn":
            x = self.norm(x, is_training)
        else:
            x = self.norm(x)
        x = jax.nn.relu(x)
        return x


# class OMLConvnet(hk.Module):
#     def __init__(
#         self,
#         hidden_size=256,
#         num_layers=6,
#         normalization="affine",
#         initializer="kaiming",
#         name=None,
#     ):
#         super().__init__(name=name)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.normalization = normalization
#         self.initializer = initializer

#         self.blocks = []
#         for stride in [1, 1, 1, 2, 1, 2]:
#             self.blocks.append(
#                 OMLBlock(hidden_size, normalization, initializer, stride)
#             )

#     def __call__(self, x, is_training):
#         for block in self.blocks:
#             x = block(x, is_training)

#         return (x,)


# class Classifier(hk.Module):
#     def __init__(
#         self,
#         output_size,
#         spatial_dims,
#         prev_hidden_size,
#         avg_pool=False,
#         initializer="kaiming",
#         head_bias=True,
#         name=None,
#     ):
#         super().__init__(name=name)
#         self.linear = hk.Linear(
#             output_size,
#             with_bias=head_bias,
#             w_init=build_initializer("relu", initializer),
#             b_init=hk.initializers.Constant(0.0),
#         )
#         self.spatial_dims = spatial_dims
#         self.prev_hidden_size = prev_hidden_size
#         self.avg_pool = avg_pool

#     def __call__(self, x, is_training=None):
#         if not self.avg_pool:
#             x = hk.Reshape((self.spatial_dims * self.prev_hidden_size,))(x)
#         x = self.linear(x)
#         return x