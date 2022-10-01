import os
import os.path as op
import time

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.bias_free_unet import unet
from learning_wavelets.models.exact_recon_old_unet import exact_recon_old_unet
from learning_wavelets.data.datasets import transform_dataset_unet


tf.random.set_seed(1)

def train_unet(cuda_visible_devices='0123', base_n_filters=64, n_layers=5):

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    # model definition
    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(0, n_layers)],
        'layers_n_non_lins': 2,
        'non_relu_contract': False,
        'bn': True,
    }


    batch_size = 32

    data_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'

    # Read Saved Batches   
    with tf.device('/CPU:0'):
        x_train = np.load(data_dir+'x_train.npy')
        y_train = np.load(data_dir+'y_train.npy')

    noise_sigma_orig = 0.0016

    # Normalisation
    def norm(arr):  
        bias = np.mean(arr, axis=(1,2), keepdims=True) #np.min(arr, axis=(1,2), keepdims=True)
        norm_fact = np.std(arr, axis=(1,2), keepdims=True) #np.max(arr, axis=(1,2), keepdims=True) - np.min(arr, axis=(1,2), keepdims=True)
        return ((arr - bias)/norm_fact), norm_fact[:,:,0,0]

    def norm2(arr):
        return arr / np.max(arr, axis=(1,2), keepdims=True)

    peak_scale_fact_tikho = ((np.max(x_train, axis=(1,2), keepdims=True) - np.min(x_train, axis=(1,2), keepdims=True)) / 
                             (np.max(y_train, axis=(1,2), keepdims=True) - np.min(y_train, axis=(1,2), keepdims=True)))

    # Normalize & scale tikho inputs
    x_train, noise_scale_fact = norm(x_train)
    x_train = norm2(x_train)
    x_train *= peak_scale_fact_tikho

    # Scale noisy sigma
    # noise_sigma_new = noise_sigma_orig / (noise_scale_fact / peak_scale_fact_tikho[:,:,0,0])

    # Normalize targets
    y_train, _ = norm(y_train)
    y_train = norm2(y_train)


    noisy_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))#.map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_noisy_ds = noisy_ds.map(
        transform_dataset_unet,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    train_noisy_ds = train_noisy_ds.batch(batch_size)
    train_noisy_ds = train_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print(train_noisy_ds)



    steps_per_epoch = np.shape(x_train)[0] // batch_size


    n_epochs = 500
    run_id = f'unet_{base_n_filters}_bias_free_{int(time.time())}'
    CHECKPOINTS_DIR = data_dir+'Trained_Models/Unet/Checkpoints/'
    chkpt_path = f'{CHECKPOINTS_DIR}/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    # callbacks preparation
    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)


    LOGS_DIR = data_dir+'Trained_Models/Unet/Logs/'
    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )

    n_channels = 1
    # run distributed
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    #     model = unet(input_size=(None, None, n_channels), lr=1e-3, **run_params)
    model = unet(input_size=(None, None, n_channels), lr=1e-3, **run_params)
    print(model.summary(line_length=114))

    # actual training
    print('\nTraining Started', flush=True)
    t0 = time.time()
    model.fit(train_noisy_ds,
              steps_per_epoch=steps_per_epoch,
              batch_size=batch_size,
              epochs=n_epochs,
            #   validation_data=val_generator,
            #   validation_steps=1,
              verbose=2,
              callbacks=[tboard_cback, chkpt_cback, lrate_cback],
              shuffle=False,)
    t1 = time.time()
    print('\nTraining Complete, Time Taken =', t1-t0, flush=True)


# if __name__ == '__main__':
#     train_unet()


def train_old_unet(cuda_visible_devices='0123',
        layers_n_non_lins=2,
        base_n_filters=128, 
        n_layers=5, 
        exact_recon=False,
    ):

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(0, n_layers)],
        'layers_n_non_lins': layers_n_non_lins,
        'non_relu_contract': False,
        'bn': False,
        'exact_recon': exact_recon,
    }

    batch_size = 32

    data_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'

    # Read Saved Batches   
    with tf.device('/CPU:0'):
        x_train = np.load(data_dir+'x_train.npy')
        y_train = np.load(data_dir+'y_train.npy')

    # noise_sigma_orig = 0.0016


    # Normalize targets
    y_train = y_train - np.mean(y_train, axis=(1,2), keepdims=True)
    norm_fact = np.max(y_train, axis=(1,2), keepdims=True) 
    y_train /= norm_fact

    # Normalize & scale tikho inputs
    x_train = x_train - np.mean(x_train, axis=(1,2), keepdims=True)
    x_train /= norm_fact

    # # Scale noisy sigma
    # noise_sigma_new = noise_sigma_orig / norm_fact



    noisy_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_noisy_ds = noisy_ds.map(
        transform_dataset_unet,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    train_noisy_ds = train_noisy_ds.batch(batch_size)
    train_noisy_ds = train_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print(train_noisy_ds)



    steps_per_epoch = np.shape(x_train)[0] // batch_size


    n_epochs = 500
    run_id = f'unet_old_exact_rec_{base_n_filters}_{int(time.time())}'
    CHECKPOINTS_DIR = data_dir+'Trained_Models/Unet/Checkpoints/'
    chkpt_path = f'{CHECKPOINTS_DIR}/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    # callbacks preparation
    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)


    LOGS_DIR = data_dir+'Trained_Models/Unet/Logs/'
    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )

    n_channels = 1
    model = exact_recon_old_unet(input_size=(None, None, n_channels), lr=1e-3, **run_params)
    print(model.summary(line_length=114))

    # actual training
    print('\nTraining Started', flush=True)
    t0 = time.time()
    model.fit(train_noisy_ds,
              steps_per_epoch=steps_per_epoch,
              batch_size=batch_size,
              epochs=n_epochs,
            #   validation_data=val_generator,
            #   validation_steps=1,
              verbose=2,
              callbacks=[tboard_cback, chkpt_cback, lrate_cback],
              shuffle=False,)
    t1 = time.time()
    print('\nTraining Complete, Time Taken =', t1-t0, flush=True)
    
    # return run_id


if __name__ == '__main__':
    train_old_unet()