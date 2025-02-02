import uuid

import chromadb
from chromadb.config import Settings
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import json
import shutil
import os

# Clean up any existing test database
persist_directory = "./test_semantic_memory"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)


class SemanticMemory:
    def __init__(self, persist_directory: str = "./memory"):
        """Initialize the semantic memory system"""
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.decay_factor = 0.98
        # Create collections with cosine similarity
        self.immediate = self.client.get_or_create_collection(
            name="immediate_memory",
            metadata={"hnsw:space": "cosine"}
        )

        self.working = self.client.get_or_create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine"}
        )

        self.long_term = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, content: str, context: Optional[Dict] = None) -> str:
        """Store new memory with optional context"""
        timestamp = datetime.now().isoformat()


        memory_id = f"mem_{uuid.uuid4()}"
        metadata = {
            "timestamp": timestamp,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "memory_id": memory_id,
            "type": "memory"
        }
        if context:
            metadata.update(context)

        # Generate embedding
        embedding = self.encoder.encode(content).tolist()

        # Store in immediate memory
        self.immediate.add(
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[memory_id]
        )

        self._consolidate_patterns()
        return memory_id



    def recall(self, query: str, n_results: int = 5, context_filter: Optional[Dict] = None, current_date: datetime = datetime.now()) -> List[Dict]:
        """Recall memories similar to query using parallel search."""
        query_embedding = self.encoder.encode(query).tolist()

        def search(collection: chromadb.Collection, mem_type):
            if collection.count() == 0:
                return []
            try:
                layer_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results * 4, collection.count()),
                    where=context_filter if context_filter and len(context_filter) > 0 else None
                )
                return self._format_results(layer_results, mem_type)
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")
                return []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(search, self.immediate, "immediate"),
                executor.submit(search, self.working, "working"),
                executor.submit(search, self.long_term, "long_term"),
            }
            results = []
            for future in futures:
                results.extend(future.result())

        # Deduplicate & sort results
        seen_contents = set()
        unique_results = []
        for result in results:
            if result['content'] not in seen_contents:
                seen_contents.add(result['content'])
                unique_results.append(result)

        unique_results.sort(key=lambda x: self.compute_memory_score(x["metadata"], x["similarity"], current_date, 1, 1), reverse=True)
        unique_results = unique_results[:n_results]
        for unique_result in unique_results:
            unique_result["metadata"]["last_access_timestamp"] = current_date.isoformat()
            unique_result["metadata"]["access_count"] = unique_result["metadata"]["access_count"] + 1
            if unique_result["metadata"]["type"] == "immediate":
                self.immediate.update(unique_result["metadata"]["memory_id"], metadatas=unique_result["metadata"])
            elif unique_result["metadata"]["type"] == "working":
                self.working.update(unique_result["metadata"]["pattern_id"], metadatas=unique_result["metadata"])
            elif unique_result["metadata"]["type"] == "long_term":
                self.long_term.update(unique_result["metadata"]["memory_id"], metadatas=unique_result["metadata"])

        return unique_results

    def _consolidate_patterns(self):
        """Consolidate patterns across memory layers"""
        immediate_memories = self.immediate.get()

        if not immediate_memories['documents']:
            return

        # Generate embeddings
        embeddings = [
            self.encoder.encode(doc)
            for doc in immediate_memories['documents']
        ]

        clusters = self._cluster_embeddings(embeddings)

        # Process immediate to working memory
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < 2:  # Skip singleton clusters
                continue


            pattern_id = f"pattern_{uuid.uuid4()}"

            pattern = self._extract_pattern(
                pattern_id,
                [immediate_memories['documents'][i] for i in cluster],
                [immediate_memories['metadatas'][i] for i in cluster]
            )

            pattern_embedding = self.encoder.encode(pattern['content']).tolist()

            # Check for duplicates
            existing = self.working.query(
                query_embeddings=[pattern_embedding],
                n_results=1
            )
            if existing['distances'] and len(existing['distances'][0]) > 0 and existing['distances'][0][0] <= 0.01:
                continue

            self.working.add(
                documents=[pattern['content']],
                metadatas=[pattern['metadata']],
                embeddings=[pattern_embedding],
                ids=[pattern_id]
            )

        # Process working to long-term memory
        working_memories = self.working.get()
        if working_memories['documents']:
            for idx, (doc, meta) in enumerate(zip(working_memories['documents'],
                                                  working_memories['metadatas'])):
                date = datetime.now()
                source_count = meta.get('source_count', 0)
                pattern_age = (date -
                               datetime.fromisoformat(meta['timestamp'])).total_seconds()

                if source_count >= 3 and pattern_age > 86400:  # 24 hours
                    long_term_id = f"long_term_{uuid.uuid4()}"
                    long_term_embedding = self.encoder.encode(doc).tolist()
                    metadata = {
                        "timestamp": date,
                        "last_access_timestamp": date,
                        "access_count": meta["access_count"],
                        "long_term": long_term_id,
                        "type": "long_term"
                    }
                    self.long_term.add(
                        documents=[doc],
                        metadatas=[metadata],
                        embeddings=[long_term_embedding],
                        ids=[long_term_id]
                    )

                    if "ids" in working_memories and working_memories["ids"]:
                        self.working.delete(ids=[working_memories["ids"][idx]])

        # Cleanup immediate memory
        if len(immediate_memories['ids']) > 100:
            oldest_ids = immediate_memories['ids'][:-100]
            self.immediate.delete(ids=oldest_ids)

    def _cluster_embeddings(self, embeddings: List[np.ndarray],
                            threshold: float = 0.8) -> List[List[int]]:
        """Cluster embeddings using cosine similarity"""
        embeddings_array = np.stack(embeddings)

        # Compute cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings_array / norms[:, np.newaxis]
        similarity_matrix = np.dot(normalized, normalized.T)

        clusters = []
        used_indices = set()

        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            cluster = [i]
            used_indices.add(i)

            for j in range(i + 1, len(embeddings)):
                if j not in used_indices and similarity_matrix[i, j] > threshold:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters

    def _extract_pattern(self, pattern_id, documents: List[str],
                         metadatas: List[Dict]) -> Dict:
        """Extract pattern from cluster of similar memories"""
        date = datetime.now().isoformat()
        pattern_metadata = {
            "type": "pattern",
            "timestamp": date,
            "pattern_id": pattern_id,
            "last_access_timestamp": date,
            "access_count": 1,
            "source_count": len(documents),
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])
        }

        return {
            "content": '\n'.join(documents),
            "metadata": pattern_metadata
        }

    def _format_results(self, results: Dict, memory_type: str) -> List[Dict]:
        """Format query results"""
        formatted = []

        if not results['documents']:
            return formatted

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = max(0, min(1, 1 - dist))  # Ensure similarity in [0,1]

            if similarity > 0:  # Only return meaningful results
                formatted.append({
                    'content': doc,
                    'metadata': meta,
                    'similarity': similarity,
                    'memory_type': memory_type
                })

        return formatted

    def compute_memory_score(
        self,
        metadata,
        relevance,
        date,
        alpha_recency,
        alpha_relevance
    ):
        recency = self.compute_recency(metadata, date)
        return (
            alpha_recency * recency
            + alpha_relevance * relevance
        )

    def compute_recency(self, metadata, date):
        decay_factor = self.decay_factor
        time_diff = date - datetime.fromisoformat(
            metadata["last_access_timestamp"]
        )
        hours_diff = time_diff.total_seconds() / 3600
        recency = decay_factor**hours_diff
        return recency

    @staticmethod
    def normalize_scores(scores):
        min_score, max_score = np.min(scores), np.max(scores)
        if min_score == max_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def get_top_indices(scores, k):
        return scores.argsort()[-k:][::-1]

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            "immediate_count": self.immediate.count(),
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count()
        }

if __name__ == "__main__":
    # Create a more complex example showing memory dynamics
    from datetime import datetime, timedelta
    import time

    # Initialize with clean state
    memory = SemanticMemory("./test_semantic_memory")

    print("\n=== Initial State ===")
    print("Adding memories about a quantum computing journey...")

    # Day 1: Initial learning
    memory.store(
        "Started learning about quantum computing basics today",
        context={"category": "learning", "topic": "quantum_computing"}
    )

    memory.store(
        "Quantum bits or qubits can be in superposition, unlike classical bits",
        context={"category": "learning", "topic": "quantum_computing"}
    )

    memory.store(
        "The sky has beautiful cirrus clouds today",
        context={"category": "observation", "topic": "weather"}
    )

    # Check state
    print("\n=== After Initial Memories ===")
    results = memory.recall("Tell me about quantum computing", n_results=5)
    for r in results:
        print(f"\nMemory: {r['content']}")
        print(f"Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")

    # Add more related memories
    memory.store(
        "Learned that qubits can exist in superposition of states",
        context={"category": "learning", "topic": "quantum_computing"}
    )

    memory.store(
        "Understanding superposition is key to quantum computing",
        context={"category": "learning", "topic": "quantum_computing"}
    )

    # Add some unrelated memories
    memory.store(
        "Had coffee with Sarah to discuss the project",
        context={"category": "social", "topic": "meeting"}
    )

    print("\n=== After More Memories ===")
    print("Quantum computing related memories:")
    results = memory.recall("What do you know about quantum computing?", n_results=5)
    for r in results:
        print(f"\nMemory: {r['content']}")
        print(f"Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")

    # Simulate passage of time for pattern formation
    print("\n=== Testing Working Memory Patterns ===")
    # Add more reinforcing memories
    memory.store(
        "The concept of superposition is fascinating in quantum computing",
        context={"category": "learning", "topic": "quantum_computing"}
    )

    memory.store(
        "In quantum computing, bits can be in multiple states simultaneously due to superposition",
        context={"category": "learning", "topic": "quantum_computing"}
    )

    print("\nPatterns around superposition concept:")
    results = memory.recall("Tell me about superposition in quantum computing", n_results=5)
    for r in results:
        print(f"\nMemory: {r['content']}")
        print(f"Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")

    # Add some weather observations to see different patterns
    memory.store(
        "Another day with beautiful cirrus clouds",
        context={"category": "observation", "topic": "weather"}
    )

    memory.store(
        "The cirrus clouds create stunning patterns in the sky",
        context={"category": "observation", "topic": "weather"}
    )

    print("\n=== Testing Pattern Separation ===")
    print("Weather-related memories:")
    results = memory.recall("What do you remember about the sky and clouds?", n_results=5)
    for r in results:
        print(f"\nMemory: {r['content']}")
        print(f"Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")

    # Get statistics about memory state
    stats = memory.get_stats()
    print("\n=== Memory System Statistics ===")
    print(f"Immediate memories: {stats['immediate_count']}")
    print(f"Working memory patterns: {stats['working_count']}")
    print(f"Long-term memories: {stats['long_term_count']}")

    # Test context filtering
    print("\n=== Testing Context Filtering ===")
    results = memory.recall(
        "What have I learned?",
        n_results=5,
        context_filter={"category": "learning"}
    )
    print("\nLearning-related memories:")
    for r in results:
        print(f"\nMemory: {r['content']}")
        print(f"Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")