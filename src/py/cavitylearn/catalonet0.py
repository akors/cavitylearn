
import tensorflow as tf
import numpy as np

DTYPE = tf.float32


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def _weight_variable(name, shape, dtype=DTYPE):
    return tf.get_variable(name, shape, dtype, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape, dtype=DTYPE):
    return tf.get_variable(name, shape, dtype, tf.constant_initializer(0.1, dtype=DTYPE))


def _convlayer(input, num_filters, layer_name, keep_prob=None, pooling=True, l2scale=0.0):

    with tf.variable_scope(layer_name, regularizer=tf.contrib.layers.l2_regularizer(l2scale)) as scope:
        kernel = _weight_variable('weights', [5, 5, 5, input.get_shape().as_list()[4], num_filters])

        variable_summaries(kernel, layer_name + '/weights')

        conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [num_filters])

        variable_summaries(biases, layer_name + '/biases')

        bias = tf.nn.bias_add(conv, biases)

        tf.histogram_summary(layer_name + '/pre_activations', bias)

        output = tf.nn.relu(bias, name=scope.name)

        if pooling:
            output = tf.nn.max_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob)

        tf.histogram_summary(layer_name + '/activations', output)

    return output


def _fc_layer(input, fc_size, layer_name, keep_prob=None, l2scale=0.0):
    # get flat input size
    dim = np.prod(input.get_shape().as_list()[1:])

    with tf.variable_scope(layer_name, regularizer=tf.contrib.layers.l2_regularizer(l2scale)) as scope:
        input_flat = tf.reshape(input, [-1, dim])

        weights = _weight_variable('weights', [dim, fc_size])
        variable_summaries(weights, layer_name + '/weights')

        biases = _bias_variable('biases', [fc_size])
        variable_summaries(biases, layer_name + '/biases')

        output = tf.matmul(input_flat, weights) + biases

        tf.histogram_summary(layer_name + '/pre_activations', output)

        output = tf.nn.relu(output, name=scope.name)

        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob)

        tf.histogram_summary(layer_name + '/activations', output)

    return output


def inference(boxes, dataconfig, p_keep_conv, p_keep_hidden, l2scale=0.0):
    prev_layer = boxes

    prev_layer = _convlayer(prev_layer, 16, "conv1", keep_prob=p_keep_conv, pooling=True, l2scale=l2scale)

    prev_layer = _convlayer(prev_layer, 32, "conv2", keep_prob=p_keep_conv, pooling=True, l2scale=l2scale)

    with tf.variable_scope('conv3_multi'):
        prev_layer = _convlayer(prev_layer, 64, "conv3_1", keep_prob=None, pooling=False, l2scale=l2scale)
        prev_layer = _convlayer(prev_layer, 64, "conv3_2", keep_prob=None, pooling=False, l2scale=l2scale)
        prev_layer = _convlayer(prev_layer, 32, "conv3_3", keep_prob=p_keep_conv, pooling=True, l2scale=l2scale)

    prev_layer = _fc_layer(prev_layer, 1024, "local4", keep_prob=p_keep_hidden, l2scale=l2scale)
    prev_layer = _fc_layer(prev_layer, 1024, "local5", keep_prob=p_keep_hidden, l2scale=l2scale)

    with tf.variable_scope('softmax_linear', regularizer=tf.contrib.layers.l2_regularizer(l2scale)) as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, dataconfig.num_classes])
        biases = _bias_variable('biases', [dataconfig.num_classes])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

    return softmax_linear


def loss(logits, labels):

    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='crossentropy')

        l = tf.reduce_mean(cross_entropy, name='crossentropy_mean')

        # get regularization
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_losses) != 0:
            l2loss = tf.add_n(reg_losses, name="l2loss")
            tf.scalar_summary(l2loss.name, l2loss)

            l = tf.add(l, l2loss, name='total_loss')

    return l


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
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 500, learnrate_decay,
                                               staircase=True, name="learning_rate")
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


def accuracy(logits, labels, k=1, name="accuracy"):

    with tf.name_scope(name):
        correct = tf.nn.in_top_k(logits, labels, k)

        # Return the number of true entries.
        acc = tf.reduce_mean(tf.cast(correct, tf.float32), name=name)
        tf.scalar_summary(name, acc)

    return acc
