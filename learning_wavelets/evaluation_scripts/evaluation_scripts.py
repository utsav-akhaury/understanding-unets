import numpy as np
import tensorflow as tf
import pickle

from learning_wavelets.models.learnlet_model import Learnlet
from learning_wavelets.models.unet import unet


tf.random.set_seed(1)

def evaluate_Learnlet(noisy,
            noise_std_test,                                  
            run_id,
            n_epochs,
            n_tiling,
            n_scales,
    ):
        
    run_params = {
        'denoising_activation': 'dynamic_soft_thresholding',
        'learnlet_analysis_kwargs':{
            'n_tiling': n_tiling,
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
        'n_scales': n_scales,
        'n_reweights_learn': 1,
        'exact_reconstruction': True,
        'clip': False,
    }

    model = Learnlet(**run_params)

    inputs = [tf.zeros((1, 128, 128, 1)), tf.zeros((1, 1))]
    model(inputs)
    model.load_weights(model_dir+f'Learnlet/Checkpoints/{run_id}-{n_epochs}.hdf5')
    
    y_pred = np.zeros((noisy.shape))
    for i in range(noisy.shape[0]):
        y_pred[i] = model.predict([np.expand_dims(noisy[i], axis=0), 
                                   np.expand_dims(np.array(noise_std_test[i]), 0).reshape(-1,1)])

    return y_pred


def evaluate_unet(noisy,                           
                run_id,
                n_epochs,
                base_n_filters, 
                n_layers=5,
                layers_n_non_lins=2,
                use_bias=False,
    ):
    
    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(0, n_layers)],
        'layers_n_non_lins': layers_n_non_lins,
        'non_relu_contract': False,
        'bn': True,
        'use_bias': use_bias,
    }
    
    n_channels = 1
    model = unet(input_size=(None, None, n_channels), **run_params)

    model.load_weights(model_dir+f'Unet/Checkpoints/{run_id}-{n_epochs}.hdf5')
     
    y_pred = np.zeros((noisy.shape))
    for i in range(noisy.shape[0]):
        y_pred[i] = model.predict(np.expand_dims(noisy[i], axis=0))

    return y_pred



dat_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'
model_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/Trained_Models/'

f = open(dat_dir+'TEST_candels_lsst_sim.pkl', 'rb')
dico = pickle.load(f)
f.close()


# Norm
noise_sigma_orig = dico['noisemap']
x_test = dico['inputs_tikho_laplacian']
y_test = dico['targets']

# Normalize targets
y_test = y_test - np.mean(y_test, axis=(1,2), keepdims=True)
norm_fact = np.max(y_test, axis=(1,2), keepdims=True) 
y_test /= norm_fact

# Normalize & scale tikho inputs
x_test = x_test - np.mean(x_test, axis=(1,2), keepdims=True)
x_test /= norm_fact

# # Scale noisy sigma
noise_sigma_new = noise_sigma_orig / norm_fact[:,:,0]

res_learnlet = np.expand_dims(np.zeros((x_test.shape)), -1)
res_unet = np.expand_dims(np.zeros((x_test.shape)), -1)


for i in range(0, x_test.shape[0], 500):

    if i+500 > x_test.shape[0]:
        ind = x_test.shape[0]
    else:
        ind = i+500

    x = x_test[i:ind]

    # Learnlet
    res_learnlet[i:ind] = evaluate_Learnlet(np.expand_dims(x, axis=-1), 
                                            noise_sigma_new,
                                            'l1_learnlet_subclassed_256__exact_reco_dynamic_soft_thresholding_1678995615-150.hdf5',
                                            150,
                                            256,
                                            5)

    # Unet
    res_unet[i:ind] = evaluate_unet(np.expand_dims(x, axis=-1), 
                                    'l1_unet_64_bias_free_1678979514',
                                    500,
                                    64)

print(res_learnlet.shape, res_unet.shape)

with open(dat_dir+f'outputs/ht_l1_learnlet_ep-150.pkl', 'wb') as f1:
    pickle.dump(res_learnlet, f1, protocol=pickle.HIGHEST_PROTOCOL)

with open(dat_dir+f'outputs/l1_unet64_ep-500.pkl', 'wb') as f1:
    pickle.dump(res_unet, f1, protocol=pickle.HIGHEST_PROTOCOL)
