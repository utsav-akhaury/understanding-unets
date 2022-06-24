import tensorflow as tf # comment if non-subclassed model

def unpack_model(init_function=None, run_params=None, run_id=None, epoch=250, **dummy_kwargs):
    model = init_function(**run_params)
    inputs = [tf.zeros((1, 32, 32, 1)), tf.zeros((1, 1))] # comment if non-subclassed model
    model(inputs) # comment if non-subclassed model
    chkpt_path = f'checkpoints/{run_id}-{epoch}.hdf5'
    model.load_weights(chkpt_path)
    return model
