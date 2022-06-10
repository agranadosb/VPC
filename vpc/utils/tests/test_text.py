from unittest import TestCase

from vpc.utils.text import text_ellipsis


class TestUtilsModel(TestCase):
    def setUp(self) -> None:
        super().setUp()
    
    def test_text_ellipsis_ellipsis(self):
        text = 'text'
        value = 2
        correct_text = 'te...'

        result = text_ellipsis(text, value)

        self.assertEqual(result, correct_text)

    def test_text_ellipsis_no_ellipsis(self):
        text = 'text'
        value = 20
        correct_text = 'text'

        result = text_ellipsis(text, value)

        self.assertEqual(result, correct_text)
