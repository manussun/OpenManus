import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import hashlib # For generating deterministic "random" embeddings if sklearn is not available

class VectorStore:
    """
    Manages text memories and their embeddings for similarity search.
    Uses scikit-learn's TfidfVectorizer and cosine_similarity for in-memory operations.
    """
    def __init__(self, embedding_model_type: str = "tfidf"):
        """
        Initializes the VectorStore.

        Args:
            embedding_model_type: Specifies the embedding model. 
                                  "tfidf" (default) or "placeholder_random".
        """
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None # Will store TF-IDF matrix or list of numpy arrays
        self._embedding_model_type = embedding_model_type

        if self._embedding_model_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
            # Try to import sklearn to check availability early
            try:
                import sklearn
                print("scikit-learn is available. Using TF-IDF for embeddings.")
            except ImportError:
                print("scikit-learn not found. Falling back to placeholder_random embeddings.")
                self._embedding_model_type = "placeholder_random"
                self.vectorizer = None # Ensure vectorizer is None if sklearn failed
        
        if self._embedding_model_type == "placeholder_random":
            # No specific initialization needed for placeholder beyond type check
            self.texts_for_placeholder: List[str] = [] # Stores original texts
            self.embeddings_for_placeholder: List[np.ndarray] = [] # Stores placeholder embeddings
            self.embedding_dim = 128 # Arbitrary dimension for placeholder embeddings
            print(f"Using placeholder_random embeddings with dimension {self.embedding_dim}.")


    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a given text using the configured model.
        For TF-IDF, this is more of a "transformation" and happens in bulk in add_memory.
        This method is conceptually for on-the-fly embedding generation if needed (e.g. for a query).
        """
        if self._embedding_model_type == "tfidf":
            if not self.vectorizer:
                raise RuntimeError("TF-IDF vectorizer not initialized. This should not happen if constructor logic is correct.")
            if not self.texts: # If no texts have been added, vectorizer isn't fit
                 # Fit with the query text temporarily to get its embedding
                 # This is a simplification. Ideally, queries are transformed based on existing corpus.
                 # Or, we only allow queries after some memories are added.
                 # For now, let's assume queries are transformed based on corpus if available, else as a single doc.
                return self.vectorizer.fit_transform([text]).toarray()[0]

            return self.vectorizer.transform([text]).toarray()[0]
        
        elif self._embedding_model_type == "placeholder_random":
            # Generate a deterministic "random" vector based on the hash of the text
            hasher = hashlib.md5(text.encode('utf-8'))
            # Use the first part of the hash to seed a random number generator for reproducibility
            seed = int(hasher.hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            return rng.rand(self.embedding_dim)
        
        else:
            raise ValueError(f"Unsupported embedding model type: {self._embedding_model_type}")

    def add_memory(self, text: str, embedding: Optional[List[float]] = None) -> None:
        """
        Adds a text memory to the store.
        If using TF-IDF, the provided 'embedding' argument is ignored as TF-IDF is corpus-dependent.
        Embeddings are (re)calculated for the entire corpus.

        Args:
            text: The text string to store.
            embedding: An optional pre-computed embedding. Ignored for TF-IDF.
                       Required if embedding_model_type is not TF-IDF and you want to provide your own.
                       If None for non-TF-IDF, an embedding will be generated.
        """
        if self._embedding_model_type == "tfidf":
            if not self.vectorizer:
                 raise RuntimeError("TF-IDF vectorizer not initialized.")
            self.texts.append(text)
            # Re-fit and transform all texts. This is inefficient for large datasets
            # but simple for an in-memory store.
            self.embeddings = self.vectorizer.fit_transform(self.texts).toarray()
            print(f"Memory added. TF-IDF embeddings recomputed. Total memories: {len(self.texts)}")

        elif self._embedding_model_type == "placeholder_random":
            self.texts_for_placeholder.append(text)
            if embedding:
                 self.embeddings_for_placeholder.append(np.array(embedding))
            else:
                 self.embeddings_for_placeholder.append(self._generate_embedding(text))
            print(f"Memory added with placeholder embedding. Total memories: {len(self.texts_for_placeholder)}")
        
        else:
            raise ValueError(f"Unsupported embedding model type: {self._embedding_model_type}")


    def retrieve_relevant_memories(self, query_text: str, k: int = 5) -> List[str]:
        """
        Retrieves the k most relevant text memories for a given query text.

        Args:
            query_text: The query text string.
            k: The number of relevant memories to retrieve.

        Returns:
            A list of the k most relevant text memories.
        """
        if self._embedding_model_type == "tfidf":
            if not self.texts or self.embeddings is None or self.embeddings.shape[0] == 0:
                return []
            if not self.vectorizer:
                 raise RuntimeError("TF-IDF vectorizer not initialized.")

            query_embedding = self.vectorizer.transform([query_text]).toarray()
            if query_embedding.shape[1] != self.embeddings.shape[1]:
                # This can happen if the query text introduces new terms not in the vocab
                # of the fitted vectorizer. A more robust handling would be to ensure
                # consistent feature space, e.g. by pre-fitting vectorizer or handling OOV.
                # For now, returning empty or logging a warning.
                print("Warning: Query text contains terms not in the existing vocabulary. Similarity search might be affected.")
                # Fallback: try to fit vectorizer with query text as well, then transform query.
                # This is not ideal as it changes the vector space.
                # A better approach might be to just ignore new terms.
                # Let's proceed with current transform, cosine_similarity handles shape mismatches if one is (1,N) and other is (M,N)
                pass


            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            # Get indices of top k similarities
            # Argsort sorts in ascending order, so we take the last k indices after sorting
            if len(similarities) < k: # If fewer similarities than k
                k = len(similarities)
            
            top_k_indices = np.argsort(similarities)[-k:][::-1] # [::-1] to make it descending
            return [self.texts[i] for i in top_k_indices]

        elif self._embedding_model_type == "placeholder_random":
            if not self.texts_for_placeholder or not self.embeddings_for_placeholder:
                return []
            
            query_embedding = self._generate_embedding(query_text)
            
            # Calculate cosine similarities
            similarities = []
            for emb in self.embeddings_for_placeholder:
                # Cosine similarity = (A . B) / (||A|| * ||B||)
                dot_product = np.dot(query_embedding, emb)
                norm_query = np.linalg.norm(query_embedding)
                norm_emb = np.linalg.norm(emb)
                if norm_query == 0 or norm_emb == 0: # Avoid division by zero
                    sim = 0.0
                else:
                    sim = dot_product / (norm_query * norm_emb)
                similarities.append(sim)
            
            similarities = np.array(similarities)

            if len(similarities) < k:
                k = len(similarities)
            
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            return [self.texts_for_placeholder[i] for i in top_k_indices]
        
        else:
            raise ValueError(f"Unsupported embedding model type: {self._embedding_model_type}")

    def get_embedding_type(self) -> str:
        """Returns the type of embedding model used."""
        return self._embedding_model_type

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Test with TF-IDF
    print("--- Testing TF-IDF VectorStore ---")
    try:
        store_tfidf = VectorStore(embedding_model_type="tfidf")
        store_tfidf.add_memory("The cat sat on the mat.")
        store_tfidf.add_memory("A dog chased the cat.")
        store_tfidf.add_memory("The mat was soft and fluffy.")
        store_tfidf.add_memory("Another dog played in the park.")

        query1 = "feline animal"
        relevant_tfidf = store_tfidf.retrieve_relevant_memories(query1, k=2)
        print(f"Memories relevant to '{query1}': {relevant_tfidf}")

        query2 = "canine pet"
        relevant_tfidf_2 = store_tfidf.retrieve_relevant_memories(query2, k=2)
        print(f"Memories relevant to '{query2}': {relevant_tfidf_2}")
    except Exception as e:
        print(f"TF-IDF test failed: {e}")

    print("\n--- Testing Placeholder_Random VectorStore ---")
    store_random = VectorStore(embedding_model_type="placeholder_random")
    store_random.add_memory("The cat sat on the mat.")
    store_random.add_memory("A dog chased the cat.")
    store_random.add_memory("The mat was soft and fluffy.")
    store_random.add_memory("Another dog played in the park.")

    query_random = "feline animal" # Query text doesn't really matter for random if not hashed
    relevant_random = store_random.retrieve_relevant_memories(query_random, k=2)
    print(f"Memories relevant to '{query_random}' (random): {relevant_random}")

    query_random_2 = "canine pet"
    relevant_random_2 = store_random.retrieve_relevant_memories(query_random_2, k=2)
    print(f"Memories relevant to '{query_random_2}' (random): {relevant_random_2}")
