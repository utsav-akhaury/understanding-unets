import os.path as op
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
#import tensorflow_addons as tfa

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.exact_recon_unet import ExactReconUnet

from ..evaluate import keras_psnr, keras_ssim


def train_unet(
        noise_std_train=(0, 55),
        noise_std_val=30,
        n_samples=None,
        source='bsd500',
        base_n_filters=4,
        n_layers=4,
        non_linearity='relu',
        batch_size=8,
        n_epochs=50,
        bn=False,
        exact_recon=False,
        residual=False,
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

    run_id = f'ExactReconUnet_{base_n_filters}_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    # callbacks preparation

    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)

    chkpt_cback = ModelCheckpoint(chkpt_path, period=min(500, n_epochs), save_weights_only=True)
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
        model = ExactReconUnet(
            n_output_channels=1, 
            kernel_size=3, 
            layers_n_channels=[base_n_filters*2**j for j in range(0, n_layers)], 
            non_linearity='relu',
            bn=bn,
            exact_recon=exact_recon,
            residual=residual,
        )
        model.compile(
        optimizer=Adam(lr=1e-3, clipnorm=1.),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
        )

    
    # actual training
    model.fit(
        im_ds_train,
        steps_per_epoch=200,
        epochs=n_epochs,
        validation_data=im_ds_val,
        validation_steps=15,
        verbose=1,
        callbacks=[tboard_cback, chkpt_cback, lrate_cback],
        shuffle=False,
    )
    
    return run_id