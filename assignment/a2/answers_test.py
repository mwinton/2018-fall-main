import hashlib
import unittest
import yaml

class AnswersTest(unittest.TestCase):
    REQUIRED_ANSWERS = {'exploration_a_1_1_positive_fraction': "<class 'float'>", 'exploration_a_1_2_balanced': "<class 'bool'>", 'exploration_a_1_3_common_class_accuracy': "<class 'float'>", 'exploration_a_2_common_tokens': "<class 'list'>", 'exploration_a_3_percentile_length': "<class 'int'>", 'exploration_a_4_problematic': "<class 'str'>", 'exploration_b_2_most_negative': "<class 'str'>", 'exploration_b_2_most_negative_score': "<class 'float'>", 'exploration_b_2_most_positive': "<class 'str'>", 'exploration_b_2_most_positive_score': "<class 'float'>", 'exploration_b_2_make_sense': "<class 'str'>", 'exploration_c_1_why_first_wrong': "<class 'str'>", 'exploration_c_1_why_second_right': "<class 'str'>", 'exploration_c_2_pattern': "<class 'str'>", 'exploration_c_2_subphrase_to_whole': "<class 'list'>", 'exploration_c_3_error_overall': "<class 'float'>", 'exploration_c_3_error_interesting': "<class 'float'>", 'exploration_c_3_error_increase': "<class 'float'>", 'bow_d_1_w_embed': "<class 'list'>", 'bow_d_1_w_0': "<class 'list'>", 'bow_d_1_b_0': "<class 'list'>", 'bow_d_1_w_1': "<class 'list'>", 'bow_d_1_b_1': "<class 'list'>", 'bow_d_1_w_out': "<class 'list'>", 'bow_d_1_b_out': "<class 'list'>", 'bow_d_2_parameters_embedding': "<class 'int'>", 'bow_d_2_parameters_hidden': "<class 'int'>", 'bow_d_2_parameters_output': "<class 'int'>", 'bow_d_3_embed_dim': "<class 'int'>", 'bow_d_3_hidden_dims': "<class 'list'>", 'bow_d_4_same_predict': "<class 'bool'>", 'bow_d_4_same_predict_why': "<class 'str'>", 'bow_f_2_interesting_accuracy': "<class 'float'>", 'bow_f_2_whole_test_accuracy': "<class 'float'>", 'bow_f_2_better_than_bayes': "<class 'bool'>", 'bow_f_2_why': "<class 'str'>", 'bow_f_3_more_training': "<class 'bool'>", 'bow_f_4_overfitting': "<class 'bool'>", 'ml_racist_1_sentiment': "<class 'float'>", 'ml_racist_2_bias_rank': "<class 'list'>", 'ml_racist_3_technique': "<class 'list'>", 'ml_debias_1_evidence': "<class 'str'>", 'ml_debias_2_table_1': "<class 'str'>", 'ml_debias_3_stages': "<class 'list'>", 'ml_debias_4': "<class 'str'>", 'ml_adversarial_1_parity': "<class 'str'>", 'ml_adversarial_2_equality': "<class 'str'>", 'ml_adversarial_3_j_lambda': "<class 'str'>"}

    def setUp(self):
        with open('answers', 'r') as f:
            self.answers = yaml.safe_load(f.read())

    def test_keys(self):
        self.assertEqual(
                sorted(AnswersTest.REQUIRED_ANSWERS.keys()),
                sorted(self.answers.keys()))

    # def test_types(self):
    #    for k, v in AnswersTest.REQUIRED_ANSWERS.items():
    #      self.assertEqual(v, str(type(self.answers[k])), msg='%s is the wrong type' % k)


if __name__ == '__main__':
    unittest.main()

