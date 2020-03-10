import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from .learnlet_model import Learnlet

class IstaLearnlet(Model):
    __name__ = 'ista_learnlet'
    def __init__(self, n_iterations, forward_operator, adjoint_operator, operator_lips_cst=None, postprocess=None, **learnlet_kwargs):
        super(IstaLearnlet, self).__init__()
        self.n_iterations = n_iterations
        self.forward_operator = forward_operator
        self.adjoint_operator = adjoint_operator
        self.postprocess = postprocess
        self.learnlet = Learnlet(**learnlet_kwargs)
        self.ista_blocks = [
            IstaLayer(
                self.learnlet,
                forward_operator,
                adjoint_operator,
                operator_lips_cst,
            )
            for i in range(self.n_iterations)
        ]

    def call(self, inputs):
        measurements, subsampling_pattern = inputs
        x = self.adjoint_operator(measurements[..., 0], subsampling_pattern)[..., None]
        for ista_block in self.ista_blocks:
            # ISTA-step
            x = ista_block([x, inputs])
        x = tf.math.abs(x)
        if self.postprocess is not None:
            x = self.postprocess(x)
        return x


class IstaLayer(Layer):
    def __init__(self, learnlet, forward_operator, adjoint_operator, operator_lips_cst=None):
        self.learnlet = learnlet
        self.forward_operator = forward_operator
        self.adjoint_operator = adjoint_operator
        if operator_lips_cst is None:
            # TODO: think of sthg better
            operator_lips_cst = 1.0
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(operator_lips_cst),
            name='alpha',
            constraint=tf.keras.constraints.NonNeg(),
        )

    def call(self, x_inputs):
        x, inputs = x_inputs
        grad = self.grad(x, inputs)
        alpha = tf.cast(self.alpha, grad.dtype)
        x = x - alpha * grad
        # learnlet transform on both the real and imaginary part
        x_real = self.prox(tf.math.real(x), self.alpha)
        x_imag = self.prox(tf.math.imag(x), self.alpha)
        x = tf.complex(x_real, x_imag)
        return x

    def grad(self, x, inputs):
        measurements, subsampling_pattern = inputs
        measurements = measurements[..., 0]
        measurements_residual = self.forward_operator(x[..., 0], subsampling_pattern) - measurements
        return self.adjoint_operator(measurements_residual, subsampling_pattern)[..., None]

    def prox(self, x, alpha):
        return self.learnlet([x, alpha])
