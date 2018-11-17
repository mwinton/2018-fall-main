import hashlib
import unittest
import yaml

class AnswersTest(unittest.TestCase):
    REQUIRED_ANSWERS = {'fb_viterbi_differ', 'can_be_zero', 'can_be_zero_reason'}

    def setUp(self):
        with open('answers', 'r') as f:
            self.answers = yaml.safe_load(f.read())

    def test_keys(self):
        self.assertEqual(
                sorted(AnswersTest.REQUIRED_ANSWERS),
                sorted(self.answers.keys()))


if __name__ == '__main__':
    unittest.main()

