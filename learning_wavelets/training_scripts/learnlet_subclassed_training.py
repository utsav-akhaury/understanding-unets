import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

from learning_wavelets.evaluate import keras_psnr, keras_ssim, center_keras_psnr
from learning_wavelets.keras_utils.normalisation import NormalisationAdjustment
from learning_wavelets.models.learnlet_model import Learnlet
from learning_wavelets.data.datasets import transform_dataset

tf.random.set_seed(1)

def train_learnlet(
        cuda_visible_devices='0123',
        denoising_activation='dynamic_soft_thresholding',
        n_filters=256,  
        exact_reco=True,
        n_reweights=1,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    run_params = {
        'denoising_activation': denoising_activation,
        'learnlet_analysis_kwargs':{
            'n_tiling': n_filters,
            'mixing_details': False,
            'skip_connection': True,
            'kernel_size': 3,
        },
        'learnlet_synthesis_kwargs': {
            'res': True,
            'kernel_size': 5,
        },
        'threshold_kwargs':{
            'noise_std_norm': True,
        },
        'n_scales': 5,    # try: 7
        'n_reweights_learn': n_reweights,
        'exact_reconstruction': exact_reco,
        'clip': False,
    }

    # data preparation
    batch_size = 32

    data_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'

    # Read Saved Batches  
    with tf.device('/CPU:0'): 
        x_train = np.load(data_dir+'x_train.npy')
        y_train = np.load(data_dir+'y_train.npy')

    noise_sigma_orig = 0.0016


    # Normalize targets
    y_train = y_train - np.mean(y_train, axis=(1,2), keepdims=True)
    norm_fact = np.max(y_train, axis=(1,2), keepdims=True) 
    y_train /= norm_fact

    # Normalize & scale tikho inputs
    x_train = x_train - np.mean(x_train, axis=(1,2), keepdims=True)
    x_train /= norm_fact

    # Scale noisy sigma
    noise_sigma_new = noise_sigma_orig / norm_fact[:,:,0,0]



    noisy_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train, noise_sigma_new))

    train_noisy_ds = noisy_ds.map(
        transform_dataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    train_noisy_ds = train_noisy_ds.batch(batch_size)
    train_noisy_ds = train_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print(train_noisy_ds)



    steps_per_epoch = np.shape(x_train)[0] // batch_size


    n_epochs = 150
    undecimated_str = '' 
    if exact_reco:
        undecimated_str += '_exact_reco'
    run_id = f'learnlet_subclassed_{n_filters}_{undecimated_str}_{denoising_activation}_{int(time.time())}'
    CHECKPOINTS_DIR = data_dir+'Trained_Models/Learnlet/Checkpoints/'
    chkpt_path = f'{CHECKPOINTS_DIR}/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)




    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)



    LOGS_DIR = data_dir+'Trained_Models/Learnlet/Logs/'
    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )
    norm_cback = NormalisationAdjustment(momentum=0.99, n_pooling=5)
    norm_cback.on_train_batch_end = norm_cback.on_batch_end


    # run distributed
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Learnlet(**run_params)

        # # Load Saved Model (if training needs to be resumed)
        # inputs = [tf.zeros((1, 32, 32, 1)), tf.zeros((1, 1))]
        # model(inputs)
        # model.load_weights(f'{CHECKPOINTS_DIR}/learnlet_subclassed_256__exact_reco_dynamic_soft_thresholding_1657446701-20.hdf5')

        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[keras_psnr, keras_ssim, center_keras_psnr],
        )

    # Training
    print('\nTraining Started', flush=True)
    t0 = time.time()
    model.fit(train_noisy_ds,
              steps_per_epoch=steps_per_epoch,
              batch_size=batch_size,
              epochs=n_epochs,
            #   validation_data=val_generator,
            #   validation_steps=1,
              verbose=2,
              callbacks=[tboard_cback, chkpt_cback, norm_cback, lrate_cback],
              shuffle=False,)
    t1 = time.time()
    print('\nTraining Complete, Time Taken =', t1-t0, flush=True)

if __name__ == '__main__':
    train_learnlet()
