import unittest

from Build_Claim_Index import resolve_api_key


class ResolveApiKeyTests(unittest.TestCase):
    def test_returns_a_clean_key_unchanged(self):
        self.assertEqual(resolve_api_key("sk-abc123DEF"), "sk-abc123DEF")

    def test_strips_the_trailing_newline_that_broke_ci(self):
        # The exact shape a web-form paste or `echo` leaves behind.
        self.assertEqual(resolve_api_key("sk-abc123\n"), "sk-abc123")
        self.assertEqual(resolve_api_key("  sk-abc123  "), "sk-abc123")
        self.assertEqual(resolve_api_key("sk-abc123\r\n"), "sk-abc123")

    def test_strips_quotes_left_by_copying_out_of_a_document(self):
        self.assertEqual(resolve_api_key('"sk-abc123"'), "sk-abc123")
        self.assertEqual(resolve_api_key("'sk-abc123'"), "sk-abc123")
        self.assertEqual(resolve_api_key("“sk-abc123”"), "sk-abc123")
        self.assertEqual(resolve_api_key(' "sk-abc123" \n'), "sk-abc123")

    def test_rejects_an_internal_illegal_character_by_position(self):
        # A smart quote inside the key cannot be salvaged by stripping, so it
        # must fail loudly rather than reach httpx as an illegal header.
        with self.assertRaises(SystemExit) as caught:
            resolve_api_key("sk-abc’123")
        message = str(caught.exception)
        self.assertIn("illegal in an HTTP header", message)
        self.assertIn("position 6", message)
        self.assertIn("U+2019", message)

    def test_rejects_an_internal_space_or_newline(self):
        for bad in ("sk-abc 123", "sk-abc\n123", "sk-abc\t123"):
            with self.assertRaises(SystemExit):
                resolve_api_key(bad)

    def test_never_echoes_the_key_in_diagnostics(self):
        secret = "sk-supersecret’value"
        with self.assertRaises(SystemExit) as caught:
            resolve_api_key(secret)
        self.assertNotIn("supersecret", str(caught.exception))

    def test_rejects_missing_and_empty_values(self):
        with self.assertRaises(SystemExit) as missing:
            resolve_api_key("")
        self.assertIn("empty", str(missing.exception))

        with self.assertRaises(SystemExit):
            resolve_api_key('""')

        with self.assertRaises(SystemExit):
            resolve_api_key("   \n  ")


if __name__ == "__main__":
    unittest.main()
