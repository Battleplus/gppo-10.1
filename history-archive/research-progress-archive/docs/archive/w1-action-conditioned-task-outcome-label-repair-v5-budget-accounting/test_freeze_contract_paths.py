import unittest

from freeze_contract import mounted_wsl_path_to_windows


class FreezeContractPathTests(unittest.TestCase):
    def test_explicit_windows_mount_preserves_unicode_and_spaces(self):
        drive, windows_path = mounted_wsl_path_to_windows(
            "/mnt/e/Z博士/research-plans/label repair v3"
        )
        self.assertEqual(drive, "e")
        self.assertEqual(windows_path, "E:\\Z博士\\research-plans\\label repair v3")

    def test_native_linux_path_is_not_misconverted_to_windows(self):
        with self.assertRaisesRegex(RuntimeError, "PACKAGE_SOURCE_NOT_ON_EXPLICIT_WINDOWS_MOUNT"):
            mounted_wsl_path_to_windows("/home/asus/label-repair-v3")


if __name__ == "__main__":
    unittest.main()
