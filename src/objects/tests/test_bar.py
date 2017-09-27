import unittest
from ..bar import Bar


class BarTests(unittest.TestCase):

    def setUp(self):
        """Perform global setup before each test."""
        pass

    def tearDown(self):
        """Perform global teardown after each test."""
        pass

    # MARK: Test Cases

    def test_000_000_ShouldFail(self):
        """Test that we can get a JSON packet for an arb definition."""
        self.fail('boilderplate test!')
