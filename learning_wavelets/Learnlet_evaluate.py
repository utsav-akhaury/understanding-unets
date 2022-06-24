import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_bsd68, im_dataset_bsd500

from learning_wavelets.models.learnlet_model import Learnlet


tf.random.set_seed(1)

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max() - gt.min())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max() - gt.min()
    )


METRIC_FUNCS = dict(
    LOSS=mse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }
        self.metric_funcs = metric_funcs

    def push(self, target, recons):
        for metric, func in self.metric_funcs.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {stddevs[name]:.4g}' for name in metric_names
        )

def evaluate_Learnlet(
        run_id = 'Learnlet_16_5_dynamic_st_bsd500_0_55_2000_1620026248-200',
        n_epochs = 500,
        n_tiling = 16,
        n_scales = 5,
        kernel_size = 5,
    ):
    
    n_volumes = 68

    run_params = {
    'denoising_activation': 'dynamic_soft_thresholding',
    'learnlet_analysis_kwargs':{
        'n_tiling': n_tiling, 
        'mixing_details': False,    
        'skip_connection': True,
        'kernel_size': kernel_size,
    },
    'learnlet_synthesis_kwargs': {
        'res': True,
        'kernel_size': kernel_size,
    },
    'threshold_kwargs':{
        'noise_std_norm': True,
    },
#     'wav_type': 'bior',
    'n_scales': n_scales,
    'n_reweights_learn': 3,
    'clip': False,
}


    data_func = im_dataset_bsd68

    batch_size = 8
    patch_size = 256
    val_set = data_func(
        mode = 'validation',
        batch_size = batch_size,
        patch_size = patch_size,
        return_noise_level = True,
    )

    val_set = val_set.take(n_volumes)

    model = Learnlet(**run_params)
    #model.built = True
    inputs = [
                tf.zeros((1, 32, 32, 1)),
                tf.zeros((1, 1)),
            ]
    model(inputs)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}.hdf5')
    
    eval_res = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total = n_volumes):
        y_pred = model.predict(x)
        eval_res.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, (list(eval_res.means().values()), list(eval_res.stddevs().values()))