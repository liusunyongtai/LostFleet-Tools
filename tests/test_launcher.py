"""Check browser launch lifecycle without opening a real browser during unit tests."""

import unittest
from unittest.mock import MagicMock, patch

from lostfleet_tools import server


class LauncherTests(unittest.TestCase):
    def test_open_browser_uses_bound_port_and_local_host(self):
        bound = MagicMock()
        bound.server_port = 8123
        bound.serve_forever.side_effect = KeyboardInterrupt
        with patch("sys.argv", ["lostfleet", "--host", "0.0.0.0", "--port", "0", "--open-browser"]), \
                patch.object(server, "EditorServer") as factory, \
                patch.object(server, "Thread") as thread:
            factory.return_value.__enter__.return_value = bound
            server.main()
            thread.assert_called_once_with(target=server.webbrowser.open,
                                           args=("http://127.0.0.1:8123/",), daemon=True)
            thread.return_value.start.assert_called_once()
            bound.serve_forever.assert_called_once()

    def test_occupied_port_does_not_open_an_unrelated_service(self):
        with patch("sys.argv", ["lostfleet", "--open-browser"]), \
                patch.object(server, "EditorServer", side_effect=OSError("port already in use")), \
                patch.object(server, "Thread") as thread:
            with self.assertRaises(OSError):
                server.main()
            thread.assert_not_called()

    def test_default_does_not_launch_browser(self):
        bound = MagicMock()
        bound.serve_forever.side_effect = KeyboardInterrupt
        with patch("sys.argv", ["lostfleet"]), patch.object(server, "EditorServer") as factory, \
                patch.object(server, "Thread") as thread:
            factory.return_value.__enter__.return_value = bound
            server.main()
            thread.assert_not_called()
