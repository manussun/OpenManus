import unittest
import threading
import time
from typing import Any, List
from app.communication.blackboard import Blackboard, HUMAN_ATTENTION_FLAGS_KEY
from app.communication.notifications import AttentionFlag

class TestBlackboard(unittest.TestCase):
    """
    Unit tests for the Blackboard class.
    """

    def setUp(self):
        """Set up for each test."""
        self.blackboard = Blackboard()
        self.callback_called = False
        self.callback_args = None

    def _test_callback(self, key: str, old_value: Any, new_value: Any):
        """A simple callback function for testing subscriptions."""
        self.callback_called = True
        self.callback_args = (key, old_value, new_value)

    def test_set_and_get_value(self):
        """Test setting and getting a simple value."""
        self.blackboard.set("test_key", "test_value")
        self.assertEqual(self.blackboard.get("test_key"), "test_value")
        self.assertIsNone(self.blackboard.get("non_existent_key"))

    def test_delete_value(self):
        """Test deleting a value."""
        self.blackboard.set("test_key_del", "test_value_del")
        self.assertEqual(self.blackboard.get("test_key_del"), "test_value_del")
        self.blackboard.delete("test_key_del")
        self.assertIsNone(self.blackboard.get("test_key_del"))

    def test_subscribe_and_notify_on_set(self):
        """Test that a subscriber is notified on 'set'."""
        self.blackboard.subscribe("notify_key", self._test_callback)
        self.blackboard.set("notify_key", "new_value")
        self.assertTrue(self.callback_called)
        self.assertIsNotNone(self.callback_args)
        self.assertEqual(self.callback_args[0], "notify_key")
        self.assertIsNone(self.callback_args[1]) # old_value should be None for a new key
        self.assertEqual(self.callback_args[2], "new_value")

    def test_subscribe_and_notify_on_delete(self):
        """Test that a subscriber is notified on 'delete'."""
        self.blackboard.set("notify_delete_key", "value_to_delete")
        self.blackboard.subscribe("notify_delete_key", self._test_callback)
        self.blackboard.delete("notify_delete_key")
        self.assertTrue(self.callback_called)
        self.assertIsNotNone(self.callback_args)
        self.assertEqual(self.callback_args[0], "notify_delete_key")
        self.assertEqual(self.callback_args[1], "value_to_delete") # old_value
        self.assertIsNone(self.callback_args[2]) # new_value should be None for delete

    def test_unsubscribe(self):
        """Test that a subscriber is not notified after unsubscribing."""
        self.blackboard.subscribe("unsub_key", self._test_callback)
        self.blackboard.unsubscribe("unsub_key", self._test_callback)
        self.blackboard.set("unsub_key", "another_value")
        self.assertFalse(self.callback_called)

    def test_multiple_subscribers(self):
        """Test multiple subscribers for the same key."""
        callback1_called = False
        callback2_called = False

        def _callback1(*args):
            nonlocal callback1_called
            callback1_called = True

        def _callback2(*args):
            nonlocal callback2_called
            callback2_called = True

        self.blackboard.subscribe("multi_sub_key", _callback1)
        self.blackboard.subscribe("multi_sub_key", _callback2)
        self.blackboard.set("multi_sub_key", "multi_value")
        self.assertTrue(callback1_called)
        self.assertTrue(callback2_called)

    def test_add_and_get_attention_flag(self):
        """Test adding and retrieving an AttentionFlag."""
        flag = AttentionFlag(agent_id="TestAgent", message="Test attention flag", urgency=1)
        self.blackboard.add_attention_flag(flag)
        flags = self.blackboard.get_attention_flags()
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].agent_id, "TestAgent")
        self.assertEqual(flags[0].message, "Test attention flag")

    def test_notify_on_add_attention_flag(self):
        """Test that subscribers to HUMAN_ATTENTION_FLAGS_KEY are notified."""
        self.blackboard.subscribe(HUMAN_ATTENTION_FLAGS_KEY, self._test_callback)
        flag1 = AttentionFlag(agent_id="Agent1", message="Flag 1")
        self.blackboard.add_attention_flag(flag1)
        
        self.assertTrue(self.callback_called)
        self.assertIsNotNone(self.callback_args)
        self.assertEqual(self.callback_args[0], HUMAN_ATTENTION_FLAGS_KEY)
        
        # old_value should be an empty list (or whatever it was before adding this flag)
        self.assertIsInstance(self.callback_args[1], list) 
        self.assertEqual(len(self.callback_args[1]), 0) # Assuming it was empty before
        
        # new_value should be a list containing flag1
        self.assertIsInstance(self.callback_args[2], list)
        self.assertEqual(len(self.callback_args[2]), 1)
        self.assertEqual(self.callback_args[2][0].message, "Flag 1")

        # Reset for next call
        self.callback_called = False
        self.callback_args = None

        flag2 = AttentionFlag(agent_id="Agent2", message="Flag 2")
        self.blackboard.add_attention_flag(flag2)
        self.assertTrue(self.callback_called)
        self.assertIsNotNone(self.callback_args)
        self.assertEqual(self.callback_args[0], HUMAN_ATTENTION_FLAGS_KEY)
        self.assertEqual(len(self.callback_args[1]), 1) # old list had flag1
        self.assertEqual(self.callback_args[1][0].message, "Flag 1")
        self.assertEqual(len(self.callback_args[2]), 2) # new list has flag1, flag2
        self.assertEqual(self.callback_args[2][1].message, "Flag 2")


    def _thread_set_values(self, num_operations):
        for i in range(num_operations):
            self.blackboard.set(f"thread_key_{i}", f"thread_value_{i}")
            time.sleep(0.001) # Small sleep to increase chance of interleaving

    def _thread_get_values(self, num_operations):
        for i in range(num_operations):
            self.blackboard.get(f"thread_key_{i}")
            time.sleep(0.001)

    def test_thread_safety_set_get(self):
        """Test basic thread safety for set and get operations."""
        num_ops = 50
        setter_thread = threading.Thread(target=self._thread_set_values, args=(num_ops,))
        getter_thread = threading.Thread(target=self._thread_get_values, args=(num_ops,))

        setter_thread.start()
        getter_thread.start()

        setter_thread.join()
        getter_thread.join()

        # Check if all values set by the setter thread are present
        for i in range(num_ops):
            self.assertEqual(self.blackboard.get(f"thread_key_{i}"), f"thread_value_{i}")
        print(f"\nBlackboard thread safety test: completed {num_ops} set/get ops per thread.")

    def test_get_attention_flags_empty(self):
        """Test get_attention_flags when no flags have been added."""
        flags = self.blackboard.get_attention_flags()
        self.assertEqual(len(flags), 0)
        self.assertIsInstance(flags, list)

    def test_get_attention_flags_returns_copy(self):
        """Test that get_attention_flags returns a copy, not the internal list."""
        flag = AttentionFlag(agent_id="CopyTestAgent", message="Test copy flag")
        self.blackboard.add_attention_flag(flag)
        
        flags1 = self.blackboard.get_attention_flags()
        self.assertEqual(len(flags1), 1)
        flags1.append(AttentionFlag(agent_id="AgentX", message="Added to copy")) # Modify the returned list
        
        flags2 = self.blackboard.get_attention_flags() # Get the list again
        self.assertEqual(len(flags2), 1) # Should still be 1, proving flags1 was a copy

    def test_delete_non_existent_key(self):
        """Test deleting a non-existent key does not raise error."""
        try:
            self.blackboard.delete("this_key_does_not_exist")
        except Exception as e:
            self.fail(f"Deleting non-existent key raised an exception: {e}")

    def test_unsubscribe_non_existent_key_or_callback(self):
        """Test unsubscribing from a non-existent key or with a non-subscribed callback."""
        # Unsubscribe from a key that was never subscribed to
        try:
            self.blackboard.unsubscribe("no_such_key_subscribed", self._test_callback)
        except Exception as e:
            self.fail(f"Unsubscribing from non-existent key raised: {e}")

        # Subscribe a callback, then try to unsubscribe a different callback from that key
        self.blackboard.subscribe("key_with_one_callback", self._test_callback)
        try:
            self.blackboard.unsubscribe("key_with_one_callback", lambda x,y,z: None) # Different callback
        except Exception as e: # Should not error, just do nothing
            self.fail(f"Unsubscribing a non-registered callback raised: {e}")


if __name__ == '__main__':
    unittest.main()
