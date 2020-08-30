import jax
from jax.random import split
from jax import numpy as jnp, grad, random
from jax.tree_util import Partial as partial

import optax as ox

from data import prepare_data, fsl_sample_and_build
from models.maml_conv import make_miniimagenet_cnn, make_params
from lib import fsl_inner_loop, outer_loop, mean_xe_and_acc_dict, batched_outer_loop

TRAIN_SIZE = 500


class MetaTrainer:
    def __init__(self, rng, cfg, experiment=None, cpu=None, device=None):
        self.cfg = cfg
        self.cpu = cpu
        self.device = device
        self.experiment = experiment
        rng, rng_params = split(rng)

        # Model
        self.body, self.head = self.prepare_model()
        (
            self.slow_params,
            self.fast_params,
            self.slow_state,
            self.fast_state,
        ) = make_params(
            rng_params,
            cfg.dataset,
            self.body.init,
            self.body.apply,
            self.head.init,
            device,
        )
        # self.prepare_params(rng_params, self.device)

        # Data
        self.prepare_data()
        self.train_loader = self.prepare_trainloader()

        # Training loops
        self.inner_opt = self.make_train_inner_opt()
        self.train_loops = FSLLoops(
            cfg.train_method,
            self.inner_opt,
            self.body.apply,
            self.head.apply,
            mean_xe_and_acc_dict,
            cfg.num_inner_steps_train,
        )
        self.outer_opt, self.lr_schedule = self.make_outer_opt()
        self.outer_opt_state = self.outer_opt.init(self.params)

    @staticmethod
    def step(
        rng,
        step_num,
        opt_state,
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        bx_spt,
        by_spt,
        bx_qry,
        by_qry,
        inner_opt,
        outer_opt,
        outer_loop,
    ):
        inner_opt_state = inner_opt.init(fast_params)
        outer_loop_kwargs = {
            "inner_opt_state": inner_opt_state,
            # "slow_params": slow_params,
            # "fast_params": fast_params,
            "slow_state": slow_state,
            "fast_state": fast_state,
        }
        partial_outer_loop = partial(
            outer_loop, **outer_loop_kwargs,
        )
        grads, (slow_state, fast_state, aux) = grad(
            partial(batched_outer_loop, partial_outer_loop=partial_outer_loop),
            (1, 2),
            has_aux=True,
        )(rng, slow_params, fast_params, bx_spt, by_spt, bx_qry, by_qry)

        updates, opt_state = outer_opt.update(
            grads, opt_state, (slow_params, fast_params)
        )
        slow_params, fast_params = ox.apply_updates((slow_params, fast_params), updates)

        return opt_state, slow_params, fast_params, slow_state, fast_state, aux

    def make_train_inner_opt(self):
        return ox.sgd(self.cfg.inner_lr)

    def make_outer_opt(self):
        lr_schedule = ox.cosine_decay_schedule(
            -self.cfg.outer_lr, self.cfg.num_outer_steps, 0.1,
        )
        return (
            ox.chain(
                ox.clip(10), ox.scale_by_adam(), ox.scale_by_schedule(lr_schedule),
            ),
            lr_schedule,
        )

    def prepare_trainloader(self):
        if self.cfg.train_method == "fsl":
            return partial(
                fsl_sample_and_build,
                images=self.train_images,
                labels=self.train_labels,
                num_tasks=self.cfg.meta_batch_size,
                way=self.cfg.way,
                spt_shot=self.cfg.shot,
                qry_shot=self.cfg.qry_shot,
                disjoint=False,
                shuffled_labels=True,
            )

    def prepare_model(self):
        if self.cfg.train_method == "fsl":
            output_size = self.cfg.way
        if self.cfg.dataset == "miniimagenet":
            max_pool = True
            spatial_dims = 25
        return make_miniimagenet_cnn(
            output_size,
            self.cfg.hidden_size,
            spatial_dims,
            max_pool,
            activation=self.cfg.activation,
            track_stats=self.cfg.track_bn_stats,
        )

    def log(self, *args, **kwargs):
        if self.experiment:
            self.experiment.log(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def prepare_data(self):
        cfg = self.cfg
        (
            train_images,
            train_labels,
            val_images,
            val_labels,
            self.preprocess_fn,
        ) = prepare_data(cfg.dataset, cfg.data_dir, self.device, cfg.prefetch_data_gpu,)
        # These are used for training
        self.train_images = train_images[:, :TRAIN_SIZE]
        self.train_labels = train_labels[:, :TRAIN_SIZE]
        # These are for supervised learning validation
        self.sup_val_images = train_images[:, TRAIN_SIZE:]
        self.sup_val_labels = train_labels[:, TRAIN_SIZE:]
        # These are for few-shot-learning and transfer learning validation
        self.val_images = val_images
        self.val_labels = val_labels
        self.log("Train data:", self.train_images.shape, self.train_labels.shape)
        self.log(
            "Supervised validation data:",
            self.sup_val_images.shape,
            self.sup_val_labels.shape,
        )
        self.log(
            "FSL and Transfer learning data:",
            self.val_images.shape,
            self.val_labels.shape,
        )

    @property
    def params(self):
        return self.slow_params, self.fast_params

    @property
    def state(self):
        return self.slow_state, self.fast_state


class FSLLoops:
    def __init__(
        self,
        method,
        inner_opt,
        slow_apply,
        fast_apply,
        loss_fn,
        num_inner_steps,
        update_inner_state=False,
    ):
        self.inner_opt = inner_opt
        self.slow_apply = slow_apply
        self.fast_apply = fast_apply
        self.loss_fn = loss_fn
        self.num_inner_steps = num_inner_steps
        self.update_inner_state = update_inner_state

        if method == "fsl":
            self.inner_loop = fsl_inner_loop
            train_outer_loop_kwargs = {
                "inner_loop": self.inner_loop,
                "is_training": True,
                "inner_opt": inner_opt,
                "num_steps": num_inner_steps,
                "slow_apply": self.slow_apply,
                "fast_apply": self.fast_apply,
                "loss_fn": self.loss_fn,
            }
            self.train_outer_loop = partial(outer_loop, **train_outer_loop_kwargs)

