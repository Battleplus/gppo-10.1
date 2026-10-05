import unittest

from registration_identity import derive_name_id, is_supported_real_name


class RegistrationIdentityTests(unittest.TestCase):
    def test_english_identity_is_stable(self):
        self.assertTrue(is_supported_real_name("libinghao"))
        self.assertEqual(derive_name_id("libinghao"), "libinghao")

    def test_chinese_identity_is_lowercase_ascii_and_stable(self):
        self.assertTrue(is_supported_real_name("李炳昊"))
        value = derive_name_id("李炳昊")
        self.assertRegex(value, r"^[a-z]+$")
        self.assertEqual(value, derive_name_id("李炳昊"))

    def test_shared_or_malformed_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHARED_ACCOUNT_NAME_NOT_ALLOWED"):
            derive_name_id("user")
        with self.assertRaisesRegex(ValueError, "REAL_NAME_FORMAT_INVALID"):
            derive_name_id("User Name")


if __name__ == "__main__":
    unittest.main()
