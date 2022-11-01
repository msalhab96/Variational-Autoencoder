from argparse import ArgumentParser


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--img_size', type=int, default=28**2
        )
    group.add_argument(
        '--hidden_size', default=512, type=int
    )
    group.add_argument(
        '--latent_size', default=16, type=int
    )
    group.add_argument(
        '--n_layers', default=4, type=int
    )
    group.add_argument(
        '--p_dropout', default=0.1, type=float
    )
    group.add_argument(
        '--conditional', default=False, action='store_true'
    )
    group.add_argument(
        '--cond_size', default=128, type=int
    )


def add_train_args(parser):
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--ckpt_dir', type=str, default='outdir/'
        )
    group.add_argument(
        '--data_path', type=str, default='data/'
        )
    group.add_argument(
        '--batch_size', type=int, default=16
        )
    group.add_argument(
        '--epochs', type=int, default=100
        )
    group.add_argument(
        '--logdir', type=str, default='outdir/logs'
        )
    group.add_argument(
        '--n_class', type=int, default=10
        )
    group.add_argument(
        '--pre_trained', type=str, default=''
        )
    group.add_argument(
        '--device', type=str, default='cuda:1'
        )
    group.add_argument(
        '--beta', type=float, default=1.0
        )


def get_cfg():
    parser = ArgumentParser()
    add_model_args(parser)
    add_train_args(parser)
    return parser.parse_args()


def get_model_args(cfg):
    return {
        'n_class': cfg.n_class,
        'cond_size': cfg.cond_size,
        'in_size': cfg.img_size,
        'hidden_size': cfg.hidden_size,
        'latent_size': cfg.latent_size,
        'n_layers': cfg.n_layers,
        'p_dropout': cfg.p_dropout
    }
