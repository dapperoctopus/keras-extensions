"""
Author: James Fiacco

Notes: Based on LSTM Layer Built into Keras and https://arxiv.org/abs/1609.07959
(c) James Fiacco 2017
"""

import numpy as np

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.legacy import interfaces


class mLSTM(Recurrent):
    """Long-Short Term Memory unit - Hochreiter 1997.
    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        multiplicative_units: Positive integer, numper of multiplicative
            units.
        multiplicative_initializer: Initializer for the `multiplicative_kernel`
            weights matrix, used for the linear transformation of the inputs.
        multiplicative_regularizer: Regularizer function applied to
            the `multiplicative_kernel` weights matrix
        multiplicative_constraint: Constraint function applied to
            the `multiplicative_kernel` weights matrix
    # References
        - [Multiplicative LSTM for sequence modelling](https://arxiv.org/pdf/1609.07959.pdf)
    """
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 multiplicative_units=None,
                 multiplicative_initializer='glorot_uniform',
                 multiplicative_regularizer=None,
                 multiplicative_constraint=None,
                 **kwargs):
        super(mLSTM, self).__init__(**kwargs)

        # Number of hidden unitsfor layer
        self.units = units

        # Outer activation function
        self.activation = activations.get(activation)

        # Internal Activation function
        self.recurrent_activation = activations.get(recurrent_activation)

        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.multiplicative_initializer = initializers.get(multiplicative_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.multiplicative_regularizer = regularizers.get(multiplicative_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.multiplicative_constraint = constraints.get(multiplicative_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        # In mLSTM, the dimension of m can be arbitrary, however we default it to being equal to the number
        # of hidden units
        if multiplicative_units:
            self.m_units = multiplicative_units
        else:
            self.m_units = self.units

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = [InputSpec(shape=(batch_size, self.units)),
                           InputSpec(shape=(batch_size, self.units))]

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight((self.input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            (self.m_units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.multiplicative_kernel = self.add_weight((self.input_dim, self.m_units),
                                                     name='multiplicative_kernel',
                                                     initializer=self.multiplicative_initializer,
                                                     regularizer=self.multiplicative_regularizer,
                                                     constraint=self.multiplicative_constraint)

        self.multiplicative_recurrent_kernel = self.add_weight((self.units, self.m_units),
                                                     name='multiplicative_recurrent_kernel',
                                                     initializer=self.recurrent_initializer,
                                                     regularizer=self.recurrent_regularizer,
                                                     constraint=self.recurrent_constraint)

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.built = True

    def preprocess_input(self, inputs, training=None):
        return inputs

    def get_constants(self, inputs, training=None):
        constants = []
        if 0. < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0. < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        x_m = K.dot(inputs * dp_mask[0], self.multiplicative_kernel)

        x_i = K.dot(inputs * dp_mask[0], self.kernel_i)
        x_f = K.dot(inputs * dp_mask[1], self.kernel_f)
        x_c = K.dot(inputs * dp_mask[2], self.kernel_c)
        x_o = K.dot(inputs * dp_mask[3], self.kernel_o)

        m = x_m * K.dot(h_tm1, self.multiplicative_recurrent_kernel)

        i = self.recurrent_activation(x_i + K.dot(m * rec_dp_mask[0],
                                                  self.recurrent_kernel_i))
        f = self.recurrent_activation(x_f + K.dot(m * rec_dp_mask[1],
                                                  self.recurrent_kernel_f))
        c = f * c_tm1 + i * (x_c + K.dot(m * rec_dp_mask[2],
                                         self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + K.dot(m * rec_dp_mask[3],
                                                  self.recurrent_kernel_o))
        h = self.activation(o * c)
        if 0. < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'multiplicative_units': self.m_units,
                  'multiplicative__initializer': initializers.serialize(self.multiplicative_initializer),
                  'multiplicative__regularizer': regularizers.serialize(self.multiplicative_regularizer),
                  'multiplicative__constraint': constraints.serialize(self.multiplicative_constraint)}

        base_config = super(mLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
