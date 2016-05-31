
import tensorflow as tf
import numpy as np

FC_SIZE = 1024
DTYPE = tf.float32


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


def inference(boxes, dataconfig, p_keep_conv, p_keep_hidden):
    prev_layer = boxes

    in_filters = dataconfig.num_props
    with tf.variable_scope('conv1') as scope:
        out_filters = 16
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        in_filters = out_filters

        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)

        prev_layer = tf.nn.relu(bias, name=scope.name)

    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    prev_layer = prev_layer  # tf.nn.lrn(prev_layer, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
    prev_layer = tf.nn.dropout(prev_layer, p_keep_conv)

    with tf.variable_scope('conv2') as scope:
        out_filters = 32

        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        in_filters = out_filters

        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)

        prev_layer = tf.nn.relu(bias, name=scope.name)

    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    prev_layer = prev_layer  # tf.nn.lrn(prev_layer, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
    prev_layer = tf.nn.dropout(prev_layer, p_keep_conv)

    with tf.variable_scope('conv3_1') as scope:
        out_filters = 64

        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        in_filters = out_filters

        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)

        prev_layer = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('conv3_2') as scope:
        out_filters = 64

        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        in_filters = out_filters

        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)

        prev_layer = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('conv3_3') as scope:
        out_filters = 32

        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        in_filters = out_filters

        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)

        prev_layer = tf.nn.relu(bias, name=scope.name)

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    prev_layer = tf.nn.dropout(prev_layer, p_keep_conv)

    with tf.variable_scope('local3') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

    prev_layer = tf.nn.dropout(local, p_keep_hidden)

    with tf.variable_scope('local4') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

    prev_layer = tf.nn.dropout(prev_layer, p_keep_hidden)

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, dataconfig.num_classes])
        biases = _bias_variable('biases', [dataconfig.num_classes])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='crossentropy')

    return tf.reduce_mean(cross_entropy, name='crossentropy_mean')


def train(loss_op, learning_rate, learnrate_decay=0.95, global_step=None):
    """Sets up the training Ops.
    Creates a summarizer to track the loss_op over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss_op: Loss tensor, from loss_op().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss_op.
    tf.scalar_summary(loss_op.op.name, loss_op)

    # decay learning rate
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 500, learnrate_decay, staircase=True, name="learning_rate")
    tf.scalar_summary(learning_rate.op.name, learning_rate)

    # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    if not global_step:
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss_op
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss_op, global_step=global_step)

    return train_op


def accuracy(logits, labels, k=1):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.

    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, labels, k)

        # Return the number of true entries.
        accuracy = tf.reduce_mean(tf.cast(correct, tf.int32), name="accuracy")
        tf.scalar_summary('accuracy', accuracy)

    return accuracy
