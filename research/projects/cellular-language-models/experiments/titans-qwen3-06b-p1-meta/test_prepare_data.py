import unittest

from prepare_data import articles


class DocumentTest(unittest.TestCase):
    def test_section_headings_remain_inside_article(self):
        lines = ["", " = Alpha = ", "first paragraph", " = = Detail = = ",
                 "second paragraph", "", " = Beta = ", "different article"]
        self.assertEqual(list(articles({"text": line} for line in lines)), [
            "= Alpha =\nfirst paragraph\n= = Detail = =\nsecond paragraph",
            "= Beta =\ndifferent article",
        ])
