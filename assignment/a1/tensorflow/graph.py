import numpy as np
import tensorflow as tf

def affine_layer(hidden_dim, x):
    '''Create an affine transformation.

    An affine transformation from linear algebra is "xW + b".

    Note that we want to compute this affine function on each
    feature vector "x" in a batch of examples and return the corresponding
    transformed vectors, each of dimension "hidden_dim".

    We'll see another way of implementing this using more sophisticated APIs
    in Assignment 2.

    Args:
      x: an op representing the features/incoming layer.
         The tensor that this op provides is of shape [batch_size x #features].
         (recall batch_size is the # of examples we want to predict in parallel)
      hidden_dim: a scalar defining the dimension of each output vector.

    Returns: a tensorflow op, when evaluated returns a tensor of dimension
             [batch_size x hidden_dim].

    Hint: On scrap paper, drop a picture of the matrix math xW + b.
    Hint: When doing the previous, make sure you draw "x" as [batch size x features]
          and the shape of the desired output as [batch_size x hidden_dim].
    Hint: use tf.get_variable to create trainable variables.
    Hint: use xavier initialization to initialize "W"
    Hint: always initialize "b" as 0s.  It isn't a constant though!
          It needs to be a trainable variable!
    '''
    pass

    # START YOUR CODE

    # Draw the sketch suggested in the hint above.
    # Include a photo of the sketch in your submission.
    # In your sketch, label all matrix/vector dimensions.

    # Create trainable variables "W" and "b"
    # Hint: use tf.get_variable, tf.zeros_initializer, and tf.contrib.layers.xavier_initializer

    # Return xW + b.
    # END YOUR CODE

def fully_connected_layers(hidden_dims, x):
    '''Construct fully connected layer(s).

    You want to construct:

    x ---> [ xW + b -> relu(.) ]* ---> output

    where the middle block is repeated 0 or more times, determined
    by the len(hidden_dims).

    Args:
      hidden_dims: A list of the width(s) of the hidden layer.
      x: a TensorFlow "op" that will evaluate to a tensor of dimension [batch_size x input_dim].

    To get the tests to pass, you must use tf.nn.relu(.) as your element-wise nonlinearity.
    
    Hint: see tf.variable_scope - you'll want to use this to make each layer 
    unique.

    Hint: a fully connected layer is a nonlinearity of an affine of its input.
          your answer here only be a couple of lines long (mine is 4).

    Hint: use your affine_layer(.) function above to construct the affine part
          of this graph.

    Hint: if hidden_dims is empty, just return x.
    '''

    # START YOUR CODE
    pass
    # END YOUR CODE


def MakeLogits(x_ph, hidden_dims):
    '''MakeLogits constructs the computation graph to turn the features into logits of the positive class.

    Args:
      x_ph: The placeholder for examples, of shape batch_size x features.
      hidden_dims: A list of sizes for the hidden layers.

    Returns:
      (A tensorflow op producing) a vector of batch size length with the logits of each example.

    Hint:  You can write this function in one line.

    Hint:  You should call fully_connected_layers to build the layers.
    
    Hint:  You should call affline_layer to turn the final hidden layer into a single scalar logit.
           (You cannot add "1" to the end of hidden_dims, because you don't want the relu.)

    Hint:  See tf.squeeze to get the final shape right: a vector, not a matrix with 1 column.
           (These look the same, but aren't.)
    
    Hint:  You will lose points if you don't provide the axis parameter to tf.squeeze.
           (Never call tf.squeeze without it!  Think about what might happen if there
           is only a single example in your batch and you omit axis!)

    Hint: Your graph should look like this:
      x ->  [Fully Connected Layer]* -> Affine Layer -> batch_size x 1 matrix -> squeeze -> Logits (batch_size vector).

    Hint: Verify the dimensions of each of your variables as you work using `print my_tensor.get_shape()`.
    Hint: Gracefully handle the case of no fully connected layers.
    Hint: The final affine layer is there to change the final output dimension
          to a scalar regardless of what the fully connected layer does.
    Hint: Just return the logits.  Don't pass it through the final sigmoid (that's done in train_nn()).
    '''
    # YOUR CODE HERE
    return None
    # END YOUR CODE HERE


def MakeLoss(logits, y_ph):
    '''MakeLoss computes the average batch cross-entropy loss.

    Args:
      logits: the logits of the positive class for each example in the batch.
      y_ph: the placeholder that represents the label for each example in the batch.

    Returns:
      (A tensorflow op producing) a scalar representing the average loss of this batch.

    Hint:  See tf.nn.sigmoid_cross_entropy_with_logits (note the dimension it expects for logits,
           this is why you needed to tf.squeeze in MakeLogits).
    Hint:  See tf.reduce_mean to go from a vector of per-item losses to the average batch-wide loss.
    '''
    # YOUR CODE HERE
    return None
    # END YOUR CODE HERE


def train_nn(X, y, X_test, hidden_dims, batch_size, num_epochs, learning_rate,
             verbose=False):
    '''
    Train a neural network consisting of fully connected layers.

    Args:
      X: train features [batch_size x features]
      Y: train labels [batch_size]
      X_test: test features [test_batch_size x features]
      hidden_dims: same as in fully_connected_layers
      learning_rate: the learning rate for your GradientDescentOptimizer.

    Returns: the predicted y label for X_test.

    The final graph should look like this:

    x ->  [Fully Connected Layer]* -> Affine Layer (scalar output, called "logits") -> Sigmoid -> y
                                                                              └------> Loss(., y)

    '''

    # Construct the placeholders.
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=[None, X.shape[-1]])
    y_ph = tf.placeholder(tf.float32, shape=[None])
    global_step = tf.Variable(0, trainable=False)

    # Construct the neural network.
    # - y_hat: probability of the positive class
    # - train_op: the training operation resulting from minimizing the loss
    #             with a GradientDescentOptimizer
    # - dimensions: you'll want to wrap the output of your final affine layer with
    #               tf.squeeze.  Make sure you specify the squeeze_dims parameter!
    #

    # Compute the logit of the positive class for each example in the batch.
    logits = MakeLogits(x_ph, hidden_dims)

    # Use those logits to compute the average batch cross-entropy loss and create an
    # optimizer to minimize this loss.
    loss = MakeLoss(logits, y_ph)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Compute the probability of the positive class.
    y_hat = tf.sigmoid(logits)

    # Output some initial statistics.
    sess = tf.Session(config=tf.ConfigProto(device_filters="/cpu:0"))
    sess.run(tf.global_variables_initializer())
    print('Initial loss:', sess.run(loss, feed_dict={x_ph: X, y_ph: y}))

    if verbose:
      for var in tf.trainable_variables():
          print('Variable: ', var.name, var.get_shape())
          print('dJ/dVar: ', sess.run(
                  tf.gradients(loss, var), feed_dict={x_ph: X, y_ph: y}))

    # Loop through your data num_epochs times...
    for epoch_num in range(num_epochs):
        # ... processing the data in batches.
        for batch in range(0, X.shape[0], batch_size):
            X_batch = X[batch : batch + batch_size]
            y_batch = y[batch : batch + batch_size]

            # Feed a batch to your network using sess.run.
            # Recall: by executing all the ops at once, TensorFlow runs the examples through
            # the network once, extracting the various tensors as needed.
            ops = [global_step, loss, train_op]
            feed_dict = {x_ph: X_batch, y_ph: y_batch}
            global_step_value, loss_value, train_op_value = sess.run(ops, feed_dict=feed_dict)

        # Dump some statistics as we train...
        if epoch_num % 300 == 0:
            print('Step: ', global_step_value, 'Loss:', loss_value)
            if verbose:
              for var in tf.trainable_variables():
                  print(var.name, sess.run(var))
              print('')

    # Return predictions.
    # There are two ways to write this...
    #
    # 1.  Evaluate P(positive class) and threshold at 0.5 (what we do below).
    # 2.  Evaluate logits and compare it against 0.0.  (Recall sigmoid(0.0) = 0.5).
    #
    # Note: If you use #2 and all you want is the most likely label (not the probability),
    # you don't ever have to explicitly compute the sigmoid.
    return 1 * (sess.run(y_hat, feed_dict={x_ph: X_test}) > 0.5)
