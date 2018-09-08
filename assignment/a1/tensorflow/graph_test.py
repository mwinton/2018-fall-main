import graph
import numpy as np
import tensorflow as tf
import unittest

class TestLayer(tf.test.TestCase):

    def test_affine(self):
        tf.set_random_seed(0)
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            z = graph.affine_layer(10, x)
            # Verify graph properties.
            self.assertAllEqual(10, z.get_shape()[-1])

            # Setup for running the graph.
            sess.run(tf.global_variables_initializer())

            # Verify that dimensions work with more than one row.
            sess.run(z, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})

            # Verify computation correct.
            x_val = np.array([[3., 2., 1.]])
            z_val = sess.run(z, feed_dict={x: x_val})
            self.assertEqual((1, 10), z_val.shape)
            self.assertAllClose([[
              -1.477905, -0.144029,  0.96745,
              0.491507, -0.105209, -0.558219,
              0.77895 ,  2.462549, -0.855641,
              2.503024]], z_val)


    def test_fully_connected_layers(self):
        tf.set_random_seed(0)
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            out = graph.fully_connected_layers([10, 20, 100, 1], x)
            self.assertAllEqual(1, out.get_shape()[-1])

            sess.run(tf.global_variables_initializer())

            sess.run(out, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})

            x_val = np.array([[-3., 2., 1.], [5., 6., 87.]])
            out_val = sess.run(out, feed_dict={x: x_val})
            self.assertEqual((2, 1), out_val.shape)
            self.assertAllClose([[0.0],[0.245194]], out_val)

    def test_fully_connected_doesnt_use_hidden_dim_as_layer_name(self):
        tf.set_random_seed(0)
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            out = graph.fully_connected_layers([10, 10, 10, 10, 20, 1], x)
            self.assertAllEqual(1, out.get_shape()[-1])
            sess.run(tf.global_variables_initializer())
            sess.run(out, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})
            x_val = np.array([[-3., 2., 1.], [5., 6., 87.]])
            out_val = sess.run(out, feed_dict={x: x_val})


    def test_no_fully_connected_layers(self):
        tf.set_random_seed(0)
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            out = graph.fully_connected_layers([], x)
            self.assertAllEqual(3, out.get_shape()[-1])

            sess.run(tf.global_variables_initializer())

            sess.run(out, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})

            x_val = np.array([[3., 2., 1.], [5., 6., 87.]])
            out_val = sess.run(out, feed_dict={x: x_val})
            self.assertEqual((2, 3), out_val.shape)
            self.assertAllClose(x_val, out_val)

    def test_make_logits(self):
        tf.set_random_seed(0)
        with self.test_session() as sess:
          x = tf.placeholder(tf.float32, shape=[None, 3])
          logits = graph.MakeLogits(x, [3, 3, 3])
          self.assertEqual(len(logits.get_shape()), 1)
          self.assertIsNone(logits.get_shape()[0].value)

          sess.run(tf.global_variables_initializer())
          out_val = sess.run(logits, feed_dict={x: [[1, 2, 3], [4, 5, 6]]})
          self.assertAllClose([1.647401, 2.838231], out_val)

    def test_make_loss(self):
       tf.set_random_seed(0)
       with self.test_session() as sess:
         logits = tf.placeholder(tf.float32, shape=[None])
         labels = tf.placeholder(tf.float32, shape=[None])
         loss = graph.MakeLoss(logits, labels)
         self.assertEqual(len(loss.get_shape()), 0)

         sess.run(tf.global_variables_initializer())
         out_val = sess.run(loss, feed_dict={logits: [3, 6, -20, 4], labels: [1, 1, 0, 1]})
         self.assertAllClose(0.017303241, out_val)



class TestNN(unittest.TestCase):

    def test_train_nn(self):
        X_train, y_train, X_test, y_test = generate_data(1000, 10)
        y_model = graph.train_nn(X_train, y_train, X_test,
                [], 50, 2, 0.001)

    def test_train_nn_with_fclayers(self):
        X_train, y_train, X_test, y_test = generate_data(1000, 10)
        y_model = graph.train_nn(X_train, y_train, X_test,
                [10, 22, 37], 50, 2, 0.001)


def generate_data(num_train, num_test):
    np.random.seed(1)
    num = num_train + num_test
    x0 = np.random.randn(num, 2) + 3.*np.array([1, 0])
    x1 = np.random.randn(num, 2) + 1.*np.array([-1, 0])
    X = np.vstack([x0, x1])
    y = np.concatenate([
        np.zeros(num), np.ones(num)])

    # Randomly shuffle the data
    shuf_idx = np.random.permutation(len(y))
    X = X[shuf_idx]
    y = y[shuf_idx]

    return X[:num_train], y[:num_train], X[num_train:], y[num_train:]


def generate_non_linear_data(num_train, num_test):
    np.random.seed(1)
    num = num_train + num_test
    x0 = np.random.randn(num, 2) + 4.*np.array([1, 0])
    x1 = np.random.randn(num, 2) + 4.*np.array([0, 1])
    x2 = np.random.randn(num, 2) + 4.*np.array([-1, 0])
    x3 = np.random.randn(num, 2) + 4.*np.array([0, -2])
    X = np.vstack([x0, x1, x2, x3])
    y = np.concatenate([
        np.zeros(num), np.ones(num),
        np.zeros(num), np.ones(num)])

    # Randomly shuffle the data
    shuf_idx = np.random.permutation(len(y))
    X = X[shuf_idx]
    y = y[shuf_idx]

    return X[:num_train], y[:num_train], X[num_train:], y[num_train:]


if __name__ == '__main__':
    unittest.main()
