import hashlib
import unittest
import yaml

class AnswersTest(unittest.TestCase):
    REQUIRED_ANSWERS = ['info_a1', 'info_a2', 'info_a2_speculation', 'info_b1_1_128msg_num_bits', 'info_b1_2_128msg_entropy', 'info_b1_3_1024msg_num_bits', 'info_b2', 'info_b3', 'info_c1', 'info_c2_1', 'info_c2_2', 'info_c3', 'info_c4', 'info_c5', 'info_c6', 'dp_a1', 'dp_a2', 'dp_a3', 'dp_b1_A', 'dp_b1_B', 'dp_b1_C', 'dp_c1', 'dp_c2', 'dp_c3', 'dp_d_helloworldhowareyou', 'dp_d_downbythebay', 'dp_d_wikipediaisareallystrongresourceontheinternet', 'dp_e1', 'dp_e2', 'dp_e3', 'dp_e4', 'tf_b_W_shape', 'tf_b_b_shape', 'tf_c_W_shape', 'tf_c_b_shape', 'tf_c_x_shape', 'tf_c_z_shape', 'tf_d_y_hat_shape', 'tf_d_y_hat_value', 'tf_d_elementwise_description', 'tf_e_W_shape', 'tf_e_b_shape', 'tf_e_x_shape', 'tf_e_z_shape']

    def setUp(self):
        with open('answers', 'r') as f:
            self.answers = yaml.safe_load(f.read())

    def test_keys(self):
        self.assertEqual(
                sorted(AnswersTest.REQUIRED_ANSWERS),
                sorted(self.answers.keys()))


if __name__ == '__main__':
    unittest.main()
