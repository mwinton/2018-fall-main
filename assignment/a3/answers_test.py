import hashlib
import unittest
import yaml

class AnswersTest(unittest.TestCase):
    REQUIRED_ANSWERS = {'a_1_distribution', 'a_1_depends_on_k', 'a_2_good_estimate', 'a_3_pqba', 'a_4_which_context', 'a_5_which_should', 'c_2_which_case', 'c_2_why', 'e_1_average_count_per_trigram_ignoring_zero_counts', 'e_1_average_count_per_trigram_including_zero_counts', 'e_2_brown', 'e_2_wikipedia', 'e_3_realistic', 'a_1_cell_func', 'a_1_parameters', 'a_2_embedding_parameters', 'a_2_output_parameters', 'a_3_single_target_word', 'a_3_full_distribution_of_all_target_words', 'a_4_with_sampled_softmax', 'a_4_with_hierarchical_softmax', 'a_5_slowest_part', 'c_1_explain_run_epoch', 'd_1_number_agreement', 'd_2_semantic_agreement', 'd_3_JJ_order'}

    def setUp(self):
        with open('answers', 'r') as f:
            self.answers = yaml.safe_load(f.read())

    def test_keys(self):
        self.assertEqual(
                sorted(AnswersTest.REQUIRED_ANSWERS),
                sorted(self.answers.keys()))


if __name__ == '__main__':
    unittest.main()

