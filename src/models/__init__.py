import haiku as hk
import jax.numpy as jnp
from jax.random import split

from .maml_conv import MiniImagenetCNNBody, MiniImagenetCNNHead
from .resnet import resnet12


def make_resnet12(test_local_stats, is_feat=False, *args, **kwargs):
    return hk.transform_with_state(
        lambda x, is_training: resnet12(*args, **kwargs,)(
            x, is_training, test_local_stats, is_feat,
        )
    )


def prepare_model(
    dataset,
    model_name,
    output_size,
    avg_pool,
    initializer,
    activation="relu",
    hidden_size=None,
    track_stats="none", # ["none", "inner", "outer", "inner", "inner-outer"]
    head_bias=True,
    norm_before_act=None,
    final_norm=False,
    normalize="bn",
    norm_init_zero=False,
    learn_residual_step=False,
):
    if dataset == "miniimagenet":
        max_pool = True
        spatial_dims = 25

    if model_name == "resnet12":
        bn_config = dict()
        if norm_init_zero:
            bn_config["scale_init"] = hk.initializers.Constant(0.)
        print("track stats", track_stats != "none")
        model = make_resnet12(
            avg_pool=True,
            w_initializer=initializer,
            activation=activation,
            test_local_stats=track_stats == "none",
            normalize=normalize,
            bn_config=bn_config,
            learn_residual_step=learn_residual_step,

        )
    elif model_name == "convnet4":
        print("track stats", track_stats != "none")

        model = hk.transform_with_state(
            lambda x, is_training: MiniImagenetCNNBody(
                hidden_size=hidden_size,
                max_pool=max_pool,
                activation=activation,
                track_stats=track_stats != "none",
                initializer=initializer,
                norm_before_act=norm_before_act,
                final_norm=final_norm,
                normalize=normalize,
                avg_pool=avg_pool,
            )(
                x, is_training,
            )
        )

    head = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNHead(
            output_size=output_size,
            # hidden_size=hidden_size,
            spatial_dims=spatial_dims,
            initializer=initializer,
            avg_pool=avg_pool,
            head_bias=head_bias,
        )(
            x, is_training,
        )
    )

    return model, head


def make_params(rng, dataset, slow_init, slow_apply, fast_init, setup_tensor):
    slow_rng, fast_rng = split(rng)
    # if dataset == "miniimagenet":
    #     setup_tensor = jnp.zeros((2, 84, 84, 3))
    # elif dataset == "omniglot":
    #     setup_tensor = jnp.zeros((2, 28, 28, 1))
    slow_params, slow_state = slow_init(slow_rng, setup_tensor, True)
    slow_outputs, _ = slow_apply(slow_params, slow_state, slow_rng, setup_tensor, True)
    fast_params, fast_state = fast_init(fast_rng, *slow_outputs, True)

    return slow_params, fast_params, slow_state, fast_state
