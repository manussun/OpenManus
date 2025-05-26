from datetime import datetime
import threading
from typing import Any, Callable, Dict, List, Optional
from app.communication.notifications import AttentionFlag # Import AttentionFlag

# Define a constant for the attention flags key
HUMAN_ATTENTION_FLAGS_KEY = "human_attention_flags"

class Blackboard:
    """
    A thread-safe blackboard system for agents to share information.
    Also manages a list of flags requiring human attention.
    """
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        # Initialize human_attention_flags directly within _data for unified handling
        self._data[HUMAN_ATTENTION_FLAGS_KEY]: List[AttentionFlag] = []
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        """
        Stores a piece of information on the blackboard.
        Notifies subscribed agents if the value changes.
        """
        with self._lock:
            old_value = self._data.get(key)
            # Special handling if the key is for attention flags to ensure it's always a list
            if key == HUMAN_ATTENTION_FLAGS_KEY and not isinstance(value, list):
                # This case should ideally not be hit if add_attention_flag is used.
                # Log a warning or error, as direct setting of this key should be controlled.
                print(f"Warning: Direct setting of '{HUMAN_ATTENTION_FLAGS_KEY}' with non-list value is discouraged. Use 'add_attention_flag'.")
                # Attempt to reconcile or raise error. For now, let's assume this won't happen with proper usage.
                # If it must be set directly, it must be a List[AttentionFlag]
                if not all(isinstance(item, AttentionFlag) for item in value):
                    raise ValueError(f"Value for '{HUMAN_ATTENTION_FLAGS_KEY}' must be a List[AttentionFlag].")

            self._data[key] = value
            if old_value != value and key in self._subscribers:
                # For attention flags, the "value" is the whole list.
                # The "old_value" is the previous list.
                for callback in self.get_subscribers(key): # Use helper to get subscribers
                    try:
                        callback(key, old_value, value)
                    except Exception as e:
                        print(f"Error in callback for key {key}: {e}")


    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a piece of information from the blackboard.
        """
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> None:
        """
        Removes a piece of information from the blackboard.
        Notifies subscribed agents that the key has been deleted.
        """
        with self._lock:
            if key in self._data:
                old_value = self._data.pop(key)
                if key in self._subscribers:
                    for callback in self._subscribers[key]:
                        try:
                            # Notify with None as the new value indicating deletion
                            callback(key, old_value, None)
                        except Exception as e:
                            # Optionally log the error
                            print(f"Error in callback for key {key} (delete): {e}")


    def subscribe(self, key: str, callback: Callable) -> None:
        """
        Allows an agent to subscribe to changes for a specific key.
        The callback will be invoked when the key's value changes or is deleted.
        The callback should accept three arguments: key, old_value, new_value.
        For deletions, new_value will be None.
        """
        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            if callback not in self._subscribers[key]:
                self._subscribers[key].append(callback)

    def unsubscribe(self, key: str, callback: Callable) -> None:
        """
        Removes a subscription for a specific key.
        """
        with self._lock:
            if key in self._subscribers and callback in self._subscribers[key]:
                self._subscribers[key].remove(callback)
                if not self._subscribers[key]: 
                    del self._subscribers[key]

    def get_subscribers(self, key: str) -> List[Callable]:
        """Safely gets subscribers for a key."""
        return self._subscribers.get(key, [])

    def add_attention_flag(self, flag: AttentionFlag) -> None:
        """
        Adds a new AttentionFlag to the list and notifies subscribers.
        """
        if not isinstance(flag, AttentionFlag):
            print(f"Error: Attempted to add non-AttentionFlag object: {flag}")
            return

        with self._lock:
            # Ensure the key exists and is a list
            if HUMAN_ATTENTION_FLAGS_KEY not in self._data or not isinstance(self._data[HUMAN_ATTENTION_FLAGS_KEY], list):
                self._data[HUMAN_ATTENTION_FLAGS_KEY] = [] # Should have been initialized, but as a safeguard
            
            current_flags = self._data[HUMAN_ATTENTION_FLAGS_KEY]
            old_flags_list_copy = list(current_flags) # Make a copy for the old_value notification

            current_flags.append(flag)
            # self._data[HUMAN_ATTENTION_FLAGS_KEY] is already updated by appending to current_flags

            # Notify subscribers for HUMAN_ATTENTION_FLAGS_KEY
            # The "value" is the new list of flags.
            # The "old_value" was the list before appending the new flag.
            for callback in self.get_subscribers(HUMAN_ATTENTION_FLAGS_KEY):
                try:
                    # Pass the copied old list and the current (new) list
                    callback(HUMAN_ATTENTION_FLAGS_KEY, old_flags_list_copy, current_flags)
                except Exception as e:
                    print(f"Error in attention flag subscriber callback: {e}")
            
            # Also, for general subscribers to any change in this specific flag object if needed (more granular)
            # For instance, if someone subscribed to the specific flag.agent_id (not typical for lists)
            # This part is more conceptual for list changes; typically subscription is to the list key itself.

    def get_attention_flags(self) -> List[AttentionFlag]:
        """
        Retrieves all current human attention flags.
        Returns a copy to prevent modification of the internal list.
        """
        with self._lock:
            # Ensure the key exists and is a list, return empty list if not properly initialized
            flags = self._data.get(HUMAN_ATTENTION_FLAGS_KEY)
            if isinstance(flags, list) and all(isinstance(item, AttentionFlag) for item in flags):
                return list(flags) # Return a copy
            # If it's not a list or not list of AttentionFlag, log error and return empty
            if flags is not None: # If it exists but is wrong type
                 print(f"Error: '{HUMAN_ATTENTION_FLAGS_KEY}' is not a list of AttentionFlag objects. Found: {type(flags)}")
            return []
