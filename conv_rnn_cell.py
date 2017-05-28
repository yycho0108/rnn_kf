class ConvLSTMCell(RNNCell):
  def __init__(self, filter_size, num_channels, forget_bias=1.0, activation=tanh, reuse=None):
    self._filter_size = filter_size # by default 3x3
    self._num_channels = num_channels
    self._forget_bias = forget_bias
    self._state_is_tuple = True 
    self._activation = activation
    self._reuse = reuse

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "conv_lstm_cell", reuse=self._reuse):
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = state
        tf.slim.conv2d(
      concat = _linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat([new_c, new_h], 1)
      return new_h, new_state

