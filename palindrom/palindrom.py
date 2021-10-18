import unittest
import re


def is_palindrom(word):
    word = word.lower()
    word = re.sub('[^a-z]', '', word)
    N = len(word)
    M = N // 2  # Middle for odd, one before middle for even
    return word[0:M] == word[N-1:N-1-M:-1]


class PalindromTest(unittest.TestCase):

    def tests(self):
        self.assertEqual(is_palindrom('Abba'), True)
        self.assertEqual(is_palindrom('Lagerregal'), True)
        self.assertEqual(is_palindrom('Reliefpfeiler'), True)
        self.assertEqual(is_palindrom('Rentner'), True)
        self.assertEqual(is_palindrom('Dienstmannamtsneid'), True)

        self.assertEqual(is_palindrom('Adam, ritt Irma da?'), True)
        self.assertEqual(is_palindrom('Eins nutzt uns: Amore. Die Rederei da, die Rederei der Omas, nutzt uns nie.'), True)

        self.assertEqual(is_palindrom('Test'), False)
        self.assertEqual(is_palindrom('Beispiel'), False)
        self.assertEqual(is_palindrom('PythonRocks'), False)
        self.assertEqual(is_palindrom('Regular Expressions sind doof!!!'), False)


if __name__ == '__main__':
    unittest.main(exit=False)
