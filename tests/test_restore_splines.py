"""Private release credentials must not follow cross-host storage redirects."""

import unittest
from urllib.request import Request

from script.restore_splines import ReleaseRedirect


class ReleaseDownloadTests(unittest.TestCase):
    def test_cross_host_redirect_removes_token(self):
        request = Request(
            "https://api.github.com/repos/example/assets/1",
            headers={"Authorization": "Bearer test-token"},
        )
        result = ReleaseRedirect().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/file",
        )
        self.assertIsNone(result.get_header("Authorization"))

    def test_same_host_redirect_preserves_access(self):
        request = Request(
            "https://api.github.com/old", headers={"Authorization": "Bearer test-token"}
        )
        result = ReleaseRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://api.github.com/new"
        )
        self.assertEqual(result.get_header("Authorization"), "Bearer test-token")
