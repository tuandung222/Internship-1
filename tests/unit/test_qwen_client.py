import unittest

from corgi.utils.parsers import _strip_think_content


class QwenClientHelpersTests(unittest.TestCase):
    def test_strip_think_content_removes_thinking_block(self):
        raw = "<think>Let me double-check the evidence.</think>The dining table features lobster and chopsticks."
        self.assertEqual(
            _strip_think_content(raw),
            "The dining table features lobster and chopsticks.",
        )

    def test_strip_think_content_handles_missing_tags(self):
        raw = "Answer without explicit think markers."
        self.assertEqual(
            _strip_think_content(raw), "Answer without explicit think markers."
        )


if __name__ == "__main__":
    unittest.main()
