import unittest
import numpy as np
from app.memory.vector_store import VectorStore

# Helper to check if sklearn is actually available in the test environment
SKLEARN_AVAILABLE = False
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

class TestVectorStore(unittest.TestCase):
    """
    Unit tests for the VectorStore class.
    """

    def common_tests_for_mode(self, store: VectorStore, mode_name: str):
        """Common tests applicable to any mode of VectorStore."""
        self.assertEqual(store.get_embedding_type(), mode_name, f"[{mode_name}] Should report correct embedding type.")

        # Test add_memory and initial state
        self.assertEqual(len(store.retrieve_relevant_memories("any query", k=3)), 0, f"[{mode_name}] Should be empty initially.")
        
        store.add_memory("memory one: apple is red")
        store.add_memory("memory two: banana is yellow")
        store.add_memory("memory three: apple and banana are fruits")

        # Test retrieve_relevant_memories
        # How to check "reasonableness" depends on the mode.
        # For TF-IDF, expect direct matches or strong overlaps.
        # For placeholder_random, consistency is key (same query = same order if no new memories).

        query1 = "apple fruit"
        relevant1_k2 = store.retrieve_relevant_memories(query1, k=2)
        self.assertEqual(len(relevant1_k2), 2, f"[{mode_name}] Should retrieve 2 memories for k=2.")

        relevant1_k5 = store.retrieve_relevant_memories(query1, k=5) # k > num_memories
        self.assertEqual(len(relevant1_k5), 3, f"[{mode_name}] Should retrieve all 3 memories if k > num_memories.")

        # Check content for TF-IDF specifically (if possible and makes sense)
        if mode_name == "tfidf" and SKLEARN_AVAILABLE:
            # With TF-IDF, "apple is red" and "apple and banana are fruits" should be most relevant
            self.assertIn("memory one: apple is red", relevant1_k2)
            self.assertIn("memory three: apple and banana are fruits", relevant1_k2)
        elif mode_name == "placeholder_random":
            # For random, we can't predict which ones, but we can check for consistency
            query_consistency = "test consistency"
            store.add_memory("consistent memory A")
            store.add_memory("consistent memory B")
            
            results_first_try = store.retrieve_relevant_memories(query_consistency, k=2)
            results_second_try = store.retrieve_relevant_memories(query_consistency, k=2)
            self.assertEqual(results_first_try, results_second_try, f"[{mode_name}] Placeholder should be deterministic for same query if memories don't change.")


        # Test _generate_embedding returns something of the correct type
        embedding_apple = store._generate_embedding("apple") # type: ignore # Accessing protected member for test
        self.assertIsInstance(embedding_apple, np.ndarray, f"[{mode_name}] _generate_embedding should return numpy array.")
        
        if mode_name == "placeholder_random":
            self.assertEqual(embedding_apple.shape, (store.embedding_dim,), f"[{mode_name}] Placeholder embedding dim mismatch.") # type: ignore
        elif mode_name == "tfidf" and SKLEARN_AVAILABLE:
            # TF-IDF embeddings depend on the fitted vocabulary.
            # If store has memories, it should have a meaningful shape.
            # If store is empty, _generate_embedding fits on the query text, so shape is (1, vocab_size_of_query)
            # This is a bit complex to assert universally without knowing vocab.
            # Just checking it's a 1D array is probably enough for _generate_embedding.
            self.assertTrue(len(embedding_apple.shape) == 1 or embedding_apple.shape[0] == 1, f"[{mode_name}] TF-IDF embedding should be 1D or row vector")


    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not available, skipping TF-IDF tests.")
    def test_tfidf_mode(self):
        """Test VectorStore in TF-IDF mode."""
        print("\nRunning VectorStore tests for TF-IDF mode...")
        store_tfidf = VectorStore(embedding_model_type="tfidf")
        self.assertEqual(store_tfidf.get_embedding_type(), "tfidf")
        self.common_tests_for_mode(store_tfidf, "tfidf")

        # Specific TF-IDF tests
        store_tfidf.add_memory("unique term_A")
        store_tfidf.add_memory("unique term_B")
        
        # Query for a term that is only in one document
        relevant_A = store_tfidf.retrieve_relevant_memories("term_A", k=1)
        self.assertEqual(len(relevant_A), 1)
        self.assertEqual(relevant_A[0], "unique term_A")

        # Test adding memory with pre-computed embedding (should be ignored by TF-IDF)
        initial_mem_count = len(store_tfidf.texts)
        store_tfidf.add_memory("new memory with ignored embedding", embedding=[0.1, 0.2, 0.3])
        self.assertEqual(len(store_tfidf.texts), initial_mem_count + 1)
        # Ensure embeddings are still TF-IDF (shape check)
        self.assertTrue(hasattr(store_tfidf, 'vectorizer'))
        self.assertIsNotNone(store_tfidf.embeddings)
        self.assertEqual(store_tfidf.embeddings.shape[0], len(store_tfidf.texts)) # type: ignore


    def test_placeholder_random_mode(self):
        """Test VectorStore in placeholder_random mode."""
        print("\nRunning VectorStore tests for placeholder_random mode...")
        store_random = VectorStore(embedding_model_type="placeholder_random")
        self.assertEqual(store_random.get_embedding_type(), "placeholder_random")
        self.common_tests_for_mode(store_random, "placeholder_random")

        # Specific placeholder_random tests
        store_random.add_memory("another placeholder text")
        embedding_dim = store_random.embedding_dim # type: ignore
        
        # Test _generate_embedding consistency
        emb1 = store_random._generate_embedding("consistent text") # type: ignore
        emb2 = store_random._generate_embedding("consistent text") # type: ignore
        np.testing.assert_array_equal(emb1, emb2, "Placeholder embeddings should be deterministic for the same text.")

        emb_diff = store_random._generate_embedding("different text for placeholder") # type: ignore
        self.assertFalse(np.array_equal(emb1, emb_diff), "Different texts should produce different placeholder embeddings.")
        self.assertEqual(emb1.shape, (embedding_dim,))

        # Test adding memory with a pre-computed embedding
        precomputed_emb = list(np.random.rand(embedding_dim))
        store_random.add_memory("memory with precomputed emb", embedding=precomputed_emb)
        
        # Check if the last added embedding matches the precomputed one
        self.assertEqual(len(store_random.texts_for_placeholder), store_random.embeddings_for_placeholder.__len__()) # type: ignore
        last_added_emb = store_random.embeddings_for_placeholder[-1] # type: ignore
        np.testing.assert_array_almost_equal(last_added_emb, np.array(precomputed_emb), decimal=6)


    def test_init_fallback_simulation(self):
        """
        Simulate fallback if TF-IDF is chosen but sklearn is not available.
        This test is more conceptual as manipulating sys.modules can be tricky and flaky.
        We mostly rely on the VectorStore's constructor logic which prints a fallback message.
        Here, we just initialize with 'tfidf' and if sklearn is NOT available,
        we expect it to behave like 'placeholder_random'.
        """
        print(f"\nSKLEARN_AVAILABLE status for fallback test: {SKLEARN_AVAILABLE}")
        store = VectorStore(embedding_model_type="tfidf") # Request TF-IDF
        if not SKLEARN_AVAILABLE:
            self.assertEqual(store.get_embedding_type(), "placeholder_random", "Should fallback to placeholder_random if sklearn is missing.")
            # Perform a simple placeholder mode check
            store.add_memory("fallback test memory")
            emb = store._generate_embedding("fallback test memory") # type: ignore
            self.assertEqual(emb.shape, (store.embedding_dim,), "Fell back, so embedding dim should match placeholder.") # type: ignore
        else:
            self.assertEqual(store.get_embedding_type(), "tfidf", "Sklearn is available, should remain tfidf.")
            # Perform a simple tfidf mode check
            store.add_memory("sklearn available test memory")
            emb = store._generate_embedding("sklearn available test memory") # type: ignore
            self.assertTrue(len(emb.shape) == 1 or emb.shape[0] == 1, "TF-IDF embedding should be 1D or row vector")


if __name__ == '__main__':
    unittest.main()
