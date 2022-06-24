import os
import os.path as op
import time

# import click
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

# from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
# from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.evaluate import keras_psnr, keras_ssim, center_keras_psnr
from learning_wavelets.keras_utils.normalisation import NormalisationAdjustment
from learning_wavelets.models.learnlet_model import Learnlet

tf.random.set_seed(1)

# @click.command()
# @click.option(
#     'noise_std_train',
#     '--ns-train',
#     nargs=2,
#     default=(0, 55),
#     type=float,
#     help='The noise standard deviation range for the training set. Defaults to [0, 55]',
# )
# @click.option(
#     'noise_std_val',
#     '--ns-val',
#     default=30,
#     type=float,
#     help='The noise standard deviation for the validation set. Defaults to 30',
# )
# @click.option(
#     'n_samples',
#     '-n',
#     default=None,
#     type=int,
#     help='The number of samples to use for training. Defaults to None, which means that all samples are used.',
# )
# @click.option(
#     'source',
#     '-s',
#     default='bsd500',
#     type=click.Choice(['bsd500', 'div2k'], case_sensitive=False),
#     help='The dataset you wish to use for training and validation, between bsd500 and div2k. Defaults to bsd500',
# )
# @click.option(
#     'cuda_visible_devices',
#     '-gpus',
#     '--cuda-visible-devices',
#     default='0123',
#     type=str,
#     help='The visible GPU devices. Defaults to 0123',
# )
# @click.option(
#     'denoising_activation',
#     '-da',
#     '--denoising-activation',
#     default='dynamic_soft_thresholding',
#     type=click.Choice([
#         'dynamic_soft_thresholding',
#         'dynamic_hard_thresholding',
#         'dynamic_soft_thresholding_per_filter',
#         'cheeky_dynamic_hard_thresholding'
#     ], case_sensitive=False),
#     help='The denoising activation to use. Defaults to dynamic_soft_thresholding',
# )
# @click.option(
#     'n_filters',
#     '-nf',
#     '--n-filters',
#     default=256,
#     type=int,
#     help='The number of filters in the learnlets. Defaults to 256.',
# )
# @click.option(
#     'decreasing_noise_level',
#     '--decr-n-lvl',
#     is_flag=True,
#     help='Set if you want the noise level distribution to be non uniform, skewed towards low value.',
# )
# @click.option(
#     'undecimated',
#     '-u',
#     is_flag=True,
#     help='Set if you want the learnlets to be undecimated.',
# )
# @click.option(
#     'exact_reco',
#     '-e',
#     is_flag=True,
#     help='Set if you want the learnlets to have exact reconstruction.',
# )
# @click.option(
#     'n_reweights',
#     '-nr',
#     default=1,
#     help='The number of reweights. Defaults to 1.',
# )
def train_learnlet(
        # noise_std_train=(0, 55),
        # noise_std_val=30,
        # n_samples=None,
        # source='bsd500',
        cuda_visible_devices='0123',
        denoising_activation='dynamic_soft_thresholding',
        n_filters=256,
        # decreasing_noise_level=True,
        # undecimated=True,
        exact_reco=True,
        n_reweights=1,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    # data preparation
    batch_size = 32

    # if source == 'bsd500':
    #     data_func = im_dataset_bsd500
    # elif source == 'div2k':
    #     data_func = im_dataset_div2k
    # im_ds_train = data_func(
    #     mode='training',
    #     batch_size=batch_size,
    #     patch_size=256,
    #     noise_std=noise_std_train,
    #     return_noise_level=True,
    #     n_samples=n_samples,
    #     decreasing_noise_level=decreasing_noise_level,
    # )
    # im_ds_val = data_func(
    #     mode='validation',
    #     batch_size=batch_size,
    #     patch_size=256,
    #     noise_std=noise_std_val,
    #     return_noise_level=True,
    # )

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
        'n_scales': 5,
        'n_reweights_learn': n_reweights,
        'exact_reconstruction': exact_reco,
        # 'undecimated': undecimated,
        'clip': False,
    }

    data_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'

    # Read Saved Batches   
    x_train = np.load(data_dir+'x_train.npy')
    y_train = np.load(data_dir+'y_train.npy')

    noise_sigma_orig = 0.0016

    # normalisation
    def norm(arr):  
        bias = np.min(arr, axis=(1,2), keepdims=True)
        norm_fact = np.max(arr, axis=(1,2), keepdims=True) - np.min(arr, axis=(1,2), keepdims=True)
        return (((arr - bias)/norm_fact) - 0.5), norm_fact

    peak_scale_fact = np.max(x_train, axis=(1,2), keepdims=True) / np.max(y_train, axis=(1,2), keepdims=True)
    
    x_train, noise_scale_fact = norm(x_train)
    x_train *= peak_scale_fact
    x_train_ds = tf.data.Dataset.from_tensor_slices(x_train).map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    noise_sigma_new = (noise_sigma_orig / noise_scale_fact)[:,:,0,0]
    noise_sigma_ds = tf.data.Dataset.from_tensor_slices(noise_sigma_new ).map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    y_train, _ = norm(y_train)
    y_train_ds = tf.data.Dataset.from_tensor_slices(x_train).map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    print(tf.shape(x_train))
    print(tf.shape(y_train))
    print(tf.shape(noise_sigma_new))

    print(x_train_ds)
    print(y_train_ds)
    print(noise_sigma_ds)




    # def add_noise_function(noise_sigma_new, y_train):

    #     def add_noise(x_train, y_train=y_train_ds, noise_sigma_new=noise_sigma_ds):

    #         def augmentation(im1, im2, noise_sigma_new):
    #             a = np.random.choice([0,1,2,3])
    #             if a==0:
    #                 return im1, im2, noise_sigma_new
    #             elif a==1:
    #                 ch = np.random.choice([1, 2, 3])
    #                 return np.rot90(im1, ch), np.rot90(im2, ch), noise_sigma_new
    #             elif a==2:
    #                 ch = np.random.choice([0, 1])
    #                 return np.flip(im1, axis=ch), np.flip(im2, axis=ch), noise_sigma_new
    #             elif a==3:
    #                 ch1 = np.random.choice([10, -10])
    #                 ch2 = np.random.choice([0, 1])
    #                 return np.roll(im1, ch1, axis=ch2), np.roll(im2, ch1, axis=ch2), noise_sigma_new

    #         def tf_augmentation(im1, im2, noise_sigma_new):
    #             # in tf
    #             im1_shape = im1.shape
    #             im2_shape = im2.shape
    #             noise_shape = noise_sigma_new.shape
    #             [im1, im2, noise_sigma_new,] = tf.py_function(augmentation, [im1, im2, noise_sigma_new], [tf.float32, tf.float32, tf.float32])
    #             #image.set_shape(im_shape)
    #             im1.set_shape(im1_shape)
    #             im2.set_shape(im2_shape)
    #             noise_sigma_new.set_shape(noise_shape)
    #             return im1, im2, noise_sigma_new

    #         x_train, y_train, noise_sigma_new = tf_augmentation(x_train, y_train, noise_sigma_new)

    #         # x_train = tf.expand_dims(x_train, axis=0)
    #         # y_train = tf.expand_dims(y_train, axis=0)
    #         # noise_sigma_new = tf.expand_dims(noise_sigma_new, axis=0)

    #         print(x_train)
    #         print(y_train)
    #         print(noise_sigma_new)

    #         return (x_train, noise_sigma_new), y_train

    #     return add_noise

    # add_noise = add_noise_function(noise_sigma_new, y_train)


    def add_noise(x_train, y_train, noise_sigma_new):

        def augmentation(im1, im2, noise_sigma_new):
            a = np.random.choice([0,1,2,3])
            if a==0:
                return im1, im2, noise_sigma_new
            elif a==1:
                ch = np.random.choice([1, 2, 3])
                return np.rot90(im1, ch), np.rot90(im2, ch), noise_sigma_new
            elif a==2:
                ch = np.random.choice([0, 1])
                return np.flip(im1, axis=ch), np.flip(im2, axis=ch), noise_sigma_new
            elif a==3:
                ch1 = np.random.choice([10, -10])
                ch2 = np.random.choice([0, 1])
                return np.roll(im1, ch1, axis=ch2), np.roll(im2, ch1, axis=ch2), noise_sigma_new

        def tf_augmentation(im1, im2, noise_sigma_new):
            # in tf
            im1_shape = im1.shape
            im2_shape = im2.shape
            noise_shape = noise_sigma_new.shape
            [im1, im2, noise_sigma_new,] = tf.py_function(augmentation, [im1, im2, noise_sigma_new], [tf.float32, tf.float32, tf.float32])
            #image.set_shape(im_shape)
            im1.set_shape(im1_shape)
            im2.set_shape(im2_shape)
            noise_sigma_new.set_shape(noise_shape)
            return im1, im2, noise_sigma_new

        x_train, y_train, noise_sigma_new = tf_augmentation(x_train, y_train, noise_sigma_new)

        print(x_train)
        print(y_train)
        print(noise_sigma_new)

        return (x_train, noise_sigma_new), y_train


    noisy_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train, noise_sigma_new))#.map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    image_noisy_ds = noisy_ds.map(
        add_noise,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    image_noisy_ds = image_noisy_ds.batch(batch_size)
    image_noisy_ds = image_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print(image_noisy_ds)






    # # Data Preprocessing Function
    # def preprocess():
    #     def preproc(image):
    #         a = np.random.choice([0,1,2,3])
    #         if a==0:
    #             return image
    #         elif a==1:
    #             return np.rot90(image, np.random.choice([1, 2, 3]))
    #         elif a==2:
    #             return np.flip(image, axis=np.random.choice([0, 1]))
    #         elif a==3:
    #             return np.roll(image, np.random.choice([10, -10]), axis=np.random.choice([0, 1]))
    #     return preproc


    # train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
    #                                                                 vertical_flip=True,
    #                                                                 dtype='float32',
    #                                                                 # validation_split=0.1,
    #                                                                 preprocessing_function=preprocess())

    # noise_datagen = tf.keras.preprocessing.image.ImageDataGenerator(dtype='float32')
    #                                                                 # validation_split=0.1)



    # def generator_multiple(x_train, y_train, noise_sigma_new, generator_im, generator_noise, batch_size):

    #     genX1 = generator_im.flow(x_train, y_train,
    #                               batch_size=batch_size,
    #                               shuffle=False, 
    #                               seed=42)

    #     genX2 = generator_noise.flow(noise_sigma_new,
    #                                  batch_size=batch_size,
    #                                  shuffle=False, 
    #                                  seed=42)

        
    #     while True:
    #         X1i = genX1.next()
    #         X2i = genX2.next()
    #         yield (X1i[0], X2i[:,:,0,0]), X1i[1]      #Yield both images and their labels
                
                
    # image_noisy_ds = generator_multiple(x_train, y_train, noise_sigma_new, train_datagen, noise_datagen, batch_size)  


    steps_per_epoch = np.shape(x_train)[0] // batch_size


    n_epochs = 100
    undecimated_str = '' #'decimated'
    # if undecimated:
    #     undecimated_str = 'un' + undecimated_str
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
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[keras_psnr, keras_ssim, center_keras_psnr],
        )

    # Training
    print('\nTraining Started', flush=True)
    t0 = time.time()
    model.fit(image_noisy_ds,
              steps_per_epoch=steps_per_epoch, #200,
              batch_size=batch_size,
              epochs=n_epochs,
            #   validation_data=val_generator,
            #   validation_steps=val_generator.samples//batch_size, #1,
              verbose=2,
              callbacks=[tboard_cback, chkpt_cback, norm_cback, lrate_cback],
              shuffle=False,)
    t1 = time.time()
    print('\nTraining Complete, Time Taken =', t1-t0, flush=True)

if __name__ == '__main__':
    train_learnlet()
