import tensorflow as tf
from tqdm import tqdm

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.evaluate import METRIC_FUNCS, Metrics 
from learning_wavelets.models.learnlet_model import Learnlet


tf.random.set_seed(1)

def evaluate_Learnlet(
        noise_std_test=30,
        run_id='Learnlet_16_5_dynamic_st_bsd500_0_55_2000_1620026248-200',
        n_epochs=500,
        n_tiling=16,
        n_scales=5,
        kernel_size=5,
        n_samples=None,
    ):
    
    noise_std_test = force_list(noise_std_test)

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

    model = Learnlet(**run_params)

    inputs = [tf.zeros((1, 32, 32, 1)), tf.zeros((1, 1))]
    model(inputs)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
    
    metrics_per_noise_level = {}

    for noise_level in noise_std_test:
        val_set = im_dataset_bsd68(
            mode='testing',
            patch_size=None,
            noise_std=noise_level,
            return_noise_level=True,
            n_samples=n_samples,
        )

        eval_res = Metrics()
        for x, y_true, size in tqdm(val_set.as_numpy_iterator()):
            y_pred = model.predict(x)
            eval_res.push(y_true, y_pred)
        metrics_per_noise_level[noise_level] = (list(eval_res.means().values()), list(eval_res.stddevs().values()))

    return METRIC_FUNCS, metrics_per_noise_level

def force_list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x
