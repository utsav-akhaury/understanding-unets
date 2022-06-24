import os.path as op
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.learnlet_model import Learnlet


tf.random.set_seed(1)


def train_Learnlet(
        noise_std_train=(0, 55), 
        noise_std_val=30,
        n_samples=None, 
        source='bsd500', 
        kernel_size=5, 
        n_tiling=16, 
        n_scales=5, 
        n_epochs=50,
        batch_size=8
    ):

    # data preparation
    if source == 'bsd500':
        data_func = im_dataset_bsd500
    elif source == 'div2k':
        data_func = im_dataset_div2k
    im_ds_train = data_func(
        mode='training',
        batch_size=batch_size,
        noise_std=noise_std_train,
        return_noise_level=True,
        n_samples=n_samples,
    )
    im_ds_val = data_func(
        mode='validation',
        batch_size=batch_size,
        noise_std=noise_std_val,
        return_noise_level=True,
    )

    run_id = f'Learnlet_{n_tiling}_{n_scales}_dynamic_st_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)
    
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

    # callbacks preparation

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=True)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )

    
    # run distributed 
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model=Learnlet(**run_params)
        model.compile(optimizer=Adam(lr=1e-3, clipnorm=0.001), loss='mse')
    

    # actual training
    model.fit(
        im_ds_train,
        steps_per_epoch=200,
        epochs=n_epochs,
        validation_data=im_ds_val,
        validation_steps=15,
        verbose=1,
        callbacks=[tboard_cback, chkpt_cback],
        shuffle=False,
    )
    
    return run_id
