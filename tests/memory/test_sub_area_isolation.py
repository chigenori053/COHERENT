import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coherent.core.memory.types import Action
from coherent.core.memory.space.store import AcceptStore, ReviewStore, RejectStore
from coherent.core.memory.space.router import MemoryRouter
from coherent.core.memory.space.layers import ProcessingResult

class TestSubAreaIsolation(unittest.TestCase):

    def setUp(self):
        self.accept_store = AcceptStore()
        self.review_store = ReviewStore()
        self.reject_store = RejectStore()
        self.router = MemoryRouter(self.accept_store, self.review_store, self.reject_store)

    def test_accept_store_storage(self):
        """Verify AcceptStore behaves normally."""
        self.accept_store.write("key1", "value1")
        result = self.accept_store.recall("key1")
        self.assertIn("key1", result)
        self.assertEqual(result["key1"], "value1")

    def test_review_store_guard(self):
        """Verify Guard-4: Review Recall Restriction."""
        self.review_store.write("review_key", "value")
        
        # Without context -> Should fail/empty
        res_no_context = self.review_store.recall("review_key", context=None)
        self.assertNotIn("review_key", res_no_context)
        
        # With wrong context -> Should fail
        res_wrong = self.review_store.recall("review_key", context="random_context")
        self.assertNotIn("review_key", res_wrong)
        
        # With valid context -> Should succeed
        res_ok = self.review_store.recall("review_key", context="human_override_review")
        self.assertIn("review_key", res_ok)

    def test_reject_store_isolation(self):
        """Verify Guard-5: Reject Store Full Isolation."""
        self.reject_store.write("bad_key", "bad_value")
        
        # Normal recall -> Should be empty
        res = self.reject_store.recall("bad_key")
        self.assertNotIn("bad_key", res)
        
        # Special context (counter_example_logging) -> Allowed 
        res_log = self.reject_store.recall("bad_key", context="counter_example_logging")
        self.assertIn("bad_key", res_log)

    def test_router_routing(self):
        """Verify MemoryRouter directs traffic correctly."""
        
        # Case 1: STORE_NEW -> Accept
        res_accept = ProcessingResult(action=Action.STORE_NEW, log=MagicMock(), hologram_ref="id_1")
        self.router.route(res_accept, "meta_data_1")
        self.assertIn("id_1", self.accept_store.recall("q"))
        
        # Case 2: REVIEW -> Review
        res_review = ProcessingResult(action=Action.REVIEW, log=MagicMock(), hologram_ref="id_2")
        self.router.route(res_review, "meta_data_2")
        # Check review store (needs context)
        self.assertIn("id_2", self.review_store.recall("q", "utility_analysis"))
        
        # Case 3: REJECT -> Reject
        res_reject = ProcessingResult(action=Action.REJECT, log=MagicMock(), hologram_ref="id_3")
        self.router.route(res_reject, "meta_data_3")
        # Check reject store (needs logging context)
        self.assertIn("id_3", self.reject_store.recall("q", "counter_example_logging"))
        # Check it is NOT in accept
        self.assertNotIn("id_3", self.accept_store.recall("q"))

if __name__ == '__main__':
    unittest.main()
