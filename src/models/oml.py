import jax
import haiku as hk
from jax._src.numpy.lax_numpy import cross
from .utils import kaiming_normal


class OMLConvnet(hk.Module):
    def __init__(
        self,
        # output_size,
        spatial_dims=9,
        image_size=84,
        conv_hidden_size=256,
        cross_replica_axis=None,
        name=None,
    ):
        super().__init__(name=name)
        self.cross_replica_axis = cross_replica_axis
        self.image_size = image_size
        self.hidden_size = conv_hidden_size
        self.spatial_dims = spatial_dims
        if image_size == 84:
            self.strides = (2, 1, 2, 1, 2, 2)
        elif image_size == 28:
            self.strides = (1, 1, 2, 1, 1, 2)

    @staticmethod
    def split_body_cls_params(params):
        return hk.data_structures.partition(
            lambda module_name, name, value: "CLS" not in module_name,
            params,
        )

    @staticmethod
    def get_train_slow_params(params):
        return OMLConvnet.split_body_cls_params(params)[0]

    @staticmethod
    def get_train_fast_params(params):
        return OMLConvnet.split_body_cls_params(params)[1]

    @staticmethod
    def get_test_slow_params(params):
        return OMLConvnet.get_train_slow_params(params)

    @staticmethod
    def get_test_fast_params(params):
        return OMLConvnet.get_train_fast_params(params)

    @staticmethod
    def get_state(state):
        # No state
        return state

    def __call__(self, x, phase="all", training=None):
        assert phase in ["all", "encoder", "adaptation"]

        if (phase == "all") or (phase == "encoder"):
            for stride in self.strides:
                x = hk.Conv2D(
                    output_channels=self.hidden_size,
                    kernel_shape=3,
                    stride=stride,
                    padding="VALID",
                    w_init=kaiming_normal,
                )(x)
                x = jax.nn.relu(x)
            x = hk.Reshape((self.hidden_size * self.spatial_dims,))(x)
        if (phase == "all") or (phase == "adaptation"):
            with hk.experimental.name_scope("CLS"):
                x = hk.Linear(
                    1000,
                    w_init=kaiming_normal,
                )(x)
        # else:
        #     x = x

        return x