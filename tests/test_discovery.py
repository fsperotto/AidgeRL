"""Discovery must match adapter support without loading heavy dependencies."""
import unittest
from unittest.mock import patch
from aidge_rl import Agent, list_packages, list_methods, register_package
from aidge_rl.agent import _PACKAGES, _PACKAGE_METHODS, _PACKAGE_MODULES


class DiscoveryTests(unittest.TestCase):
    """Exercise supported, installed, and extension catalog behavior."""

    def test_catalog_and_no_imports(self):
        """Cataloging works even when no supported backend is installed."""
        with patch('importlib.util.find_spec', return_value=None):
            self.assertIn('sb3', list_packages())
            self.assertEqual(list_packages(installed_only=True), [])
            self.assertEqual(list_methods('SB3', installed_only=True), [])
            self.assertIn('PPO', list_methods('SB3'))
        with patch('importlib.util.find_spec', return_value=object()):
            self.assertIn('sb3', list_packages(installed_only=True))
        self.assertEqual(list_methods()['sb3'], list_methods('sb3'))
        self.assertNotIn('cleanrl', list_packages())
        with self.assertRaises(ValueError):
            Agent(package='cleanrl', method='PPO')

    def test_custom_metadata_and_defensive_results(self):
        """Custom registries expose declared methods without leaking mutable state."""
        name = 'test_discovery_backend'
        try:
            register_package(name, lambda **kw: kw, methods=['Z', 'A', 'A'])
            self.assertEqual(list_methods(name), ['A', 'Z'])
            methods = list_methods(name)
            methods.clear()
            self.assertEqual(list_methods(name), ['A', 'Z'])
            self.assertNotIn(name, list_packages(installed_only=True))
            with self.assertRaises(ValueError):
                register_package(name, lambda **kw: kw)
        finally:
            for registry in (_PACKAGES, _PACKAGE_METHODS, _PACKAGE_MODULES):
                registry.pop(name, None)
        with self.assertRaises(ValueError):
            list_methods('unknown')
