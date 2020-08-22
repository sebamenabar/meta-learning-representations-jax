from tqdm import tqdm
import functools
import pickle
import numpy as onp

import jax
from jax.random import split
from jax.image import resize as im_resize
from jax.tree_util import Partial as partial
from jax import ops, nn, jit, grad, value_and_grad, lax, vmap, random, numpy as jnp

from jax.experimental import stax
from jax.experimental import optix
from jax.experimental import optimizers

import haiku as hk

from lib import (
    setup_device,
    mean_xe_and_acc,
    make_fsl_inner_outer_loop,
    make_batched_outer_loop,
)
from models.maml_conv import MiniImagenetCNNMaker
from data import prepare_miniimagenet_data, fsl_sample_tasks
from data.sampling import fsl_sample_transfer_and_build


if __name__ == "__main__":
    cpu, gpu = setup_device()
    rng = random.PRNGKey(0)
    miniimagenet_data_dir = "/workspace1/samenabar/data/mini-imagenet/"
    num_tasks = 25
    way = 5
    spt_shot = 1
    qry_shot = 15
    inner_lr = 5e-1
    outer_lr = 1e-2
    num_inner_steps = 5
    num_outer_steps = 10000
    disjoint_tasks = False

    ### DATA
    train_images, train_labels, _ = prepare_miniimagenet_data(
        miniimagenet_data_dir, "train"
    )
    val_images, val_labels, _ = prepare_miniimagenet_data(miniimagenet_data_dir, "val")
    train_images = jax.device_put(train_images, cpu)
    train_labels = jax.device_put(train_labels, cpu)
    val_images = jax.device_put(val_images, cpu)
    val_labels = jax.device_put(val_labels, cpu)

    mean = jax.device_put(jnp.array([0.4707837, 0.4494574, 0.4026407]), gpu)
    std = jax.device_put(jnp.array([0.28429058, 0.27527657, 0.29029518]), gpu)

    print("Train data:", train_images.shape, train_labels.shape)
    print("Val data:", val_images.shape, val_labels.shape)

    ### MODEL
    MiniImagenetCNN = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNMaker(output_size=way,)(x, is_training),
    )
    inner_opt = optix.chain(optix.sgd(inner_lr))

    ### FUNCTIONS
    loss_fn = mean_xe_and_acc

    def apply_and_loss_fn(
        rng, slow_params, fast_params, state, is_training, inputs, targets
    ):
        params = hk.data_structures.merge(slow_params, fast_params)
        logits, state = MiniImagenetCNN.apply(params, state, rng, inputs, is_training)
        loss, acc = loss_fn(logits, targets)
        return loss, (state, {"acc": acc})

        return logits, state

    inner_loop, outer_loop = make_fsl_inner_outer_loop(
        apply_and_loss_fn, inner_opt.update, num_inner_steps, update_state=False
    )

    batched_outer_loop = make_batched_outer_loop(outer_loop)

    ### PREPARE PARAMETERS
    params, state = MiniImagenetCNN.init(rng, jnp.zeros((2, 84, 84, 3)), False)
    params = jax.tree_map(lambda x: jax.device_put(x, gpu), params)
    state = jax.tree_map(lambda x: jax.device_put(x, gpu), state)
    predicate = lambda m, n, v: m == "mini_imagenet_cnn/linear"

    outer_opt_init, outer_opt_update, outer_get_params = optimizers.adam(
        step_size=outer_lr,
    )
    outer_opt_state = outer_opt_init(params)

    ### TRAIN FUNCTIONS
    def step(step_num, outer_opt_state, state, x_spt, y_spt, x_qry, y_qry):
        params = outer_get_params(outer_opt_state)
        fast_params, slow_params = hk.data_structures.partition(predicate, params)
        inner_opt_state = inner_opt.init(fast_params)

        (outer_loss, (state, info)), grads = value_and_grad(
            batched_outer_loop, (1, 2), has_aux=True
        )(
            None,
            slow_params,
            fast_params,
            state,
            inner_opt_state,
            True,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
        )

        grads = hk.data_structures.merge(*grads)
        outer_opt_state = outer_opt_update(i, grads, outer_opt_state)

        return outer_opt_state, state, info

    step = jit(step)
    pbar = tqdm(range(num_outer_steps))
    for i in pbar:
        rng, rng_sample = split(rng)
        x_spt, y_spt, x_qry, y_qry = fsl_sample_transfer_and_build(
            rng_sample,
            mean,
            std,
            train_images,
            train_labels,
            num_tasks,
            way,
            spt_shot,
            qry_shot,
            gpu,
            disjoint_tasks,
        )
        # sampled_images, sampled_labels = fsl_sample_tasks(
        #     rng_sample,
        #     train_images,
        #     train_labels,
        #     num_tasks=num_tasks,
        #     way=way,
        #     shot=spt_shot + qry_shot,
        #     disjoint=disjoint_tasks,
        # )
        # shuffled_labels = (
        #     jnp.repeat(jnp.arange(way), spt_shot + qry_shot)
        #     .reshape(way, spt_shot + qry_shot)[None, :]
        #     .repeat(num_tasks, 0)
        # )

        # # Transfer ints but operate on gpu
        # sampled_images = jax.device_put(sampled_images, gpu)
        # shuffled_labels = jax.device_put(shuffled_labels, gpu)

        # images_shape = sampled_images.shape[3:]
        # sampled_images = ((sampled_images / 255) - mean) / std

        # # Transfer floats but operate on cpu
        # # sampled_images = jax.device_put(sampled_images, gpu)
        # # shuffled_labels = jax.device_put(shuffled_labels, gpu)

        # x_spt, x_qry = jnp.split(sampled_images, (spt_shot,), 2)
        # x_spt = x_spt.reshape(num_tasks, way * spt_shot, *images_shape)
        # x_qry = x_qry.reshape(num_tasks, way * qry_shot, *images_shape)
        # y_spt, y_qry = jnp.split(shuffled_labels, (spt_shot,), 2)
        # y_spt = y_spt.reshape(num_tasks, way * spt_shot)
        # y_qry = y_qry.reshape(num_tasks, way * qry_shot)
        # y_spt = y_qry = shuffled_labels.reshape(num_tasks, way * shot)

        outer_opt_state, state, info = step(
            i, outer_opt_state, state, x_spt, y_spt, x_qry, y_qry
        )

        # print(info)

        if (i % 50) == 0:
            if (i % 200) == 0:
                rng, rng_sample = split(rng)
                x_spt, y_spt, x_qry, y_qry = fsl_sample_transfer_and_build(
                    rng_sample,
                    mean,
                    std,
                    val_images,
                    val_labels,
                    num_tasks,
                    way,
                    spt_shot,
                    qry_shot,
                    gpu,
                    False,
                )

                params = outer_get_params(outer_opt_state)
                fast_params, slow_params = hk.data_structures.partition(predicate, params)
                inner_opt_state = inner_opt.init(fast_params)
                val_outer_loss, (val_state, val_info) = batched_outer_loop(
                    None,
                    slow_params,
                    fast_params,
                    state,
                    inner_opt_state,
                    False,
                    x_spt,
                    y_spt,
                    x_qry,
                    y_qry,
                )

            pbar.set_postfix(
                # iol=f"{info['outer']['initial_loss'].mean():.3f}",
                loss=f"{info['outer']['final_loss'].mean():.3f}",
                # iil=f"{info['inner']['initial_loss'].mean():.3f}",
                # fil=f"{info['inner']['final_loss'].mean():.3f}",
                # iia=f"{info['inner']['initial_aux'][0]['acc'].mean():.3f}",
                # fia=f"{info['inner']['final_aux'][0]['acc'].mean():.3f}",
                # ioa=f"{info['outer']['initial_aux'][0]['acc'].mean():.3f}",
                foa=f"{info['outer']['final_aux'][0]['acc'].mean():.3f}",

                vfol=f"{val_outer_loss:.3f}",
                vioa=f"{val_info['outer']['initial_aux'][0]['acc'].mean():.3f}",
                vfoa=f"{val_info['outer']['final_aux'][0]['acc'].mean():.3f}",


            )
