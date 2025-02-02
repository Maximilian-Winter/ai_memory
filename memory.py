import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
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

        metadata = {
            "timestamp": timestamp,
            "type": "memory"
        }
        if context:
            metadata.update(context)

        memory_id = f"mem_{timestamp}"

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

    def recall(self, query: str, n_results: int = 5,
               context_filter: Optional[Dict] = None) -> List[Dict]:
        """Recall memories similar to query"""
        query_embedding = self.encoder.encode(query).tolist()
        results = []
        seen_contents = set()

        for collection, mem_type in [
            (self.immediate, "immediate"),
            (self.working, "working"),
            (self.long_term, "long_term")
        ]:
            try:
                if collection.count() == 0:
                    continue

                layer_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, collection.count()),
                    where=context_filter if context_filter and len(context_filter) > 0 else None
                )

                results.extend(self._format_results(layer_results, mem_type))
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")

        # Deduplicate and sort results
        unique_results = []
        for result in results:
            if result['content'] not in seen_contents:
                seen_contents.add(result['content'])
                unique_results.append(result)

        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        return unique_results[:n_results]

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

            pattern = self._extract_pattern(
                [immediate_memories['documents'][i] for i in cluster],
                [immediate_memories['metadatas'][i] for i in cluster]
            )

            pattern_id = f"pattern_{datetime.now().isoformat()}_{cluster_idx}"
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
                source_count = meta.get('source_count', 0)
                pattern_age = (datetime.now() -
                               datetime.fromisoformat(meta['timestamp'])).total_seconds()

                if source_count >= 3 and pattern_age > 86400:  # 24 hours
                    pattern_id = f"longterm_{datetime.now().isoformat()}_{idx}"
                    pattern_embedding = self.encoder.encode(doc).tolist()

                    self.long_term.add(
                        documents=[doc],
                        metadatas=[meta],
                        embeddings=[pattern_embedding],
                        ids=[pattern_id]
                    )

                    self.working.delete(ids=[working_memories['ids'][idx]])

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

    def _extract_pattern(self, documents: List[str],
                         metadatas: List[Dict]) -> Dict:
        """Extract pattern from cluster of similar memories"""
        latest_idx = max(range(len(metadatas)),
                         key=lambda i: metadatas[i]['timestamp'])

        pattern_metadata = {
            "type": "pattern",
            "timestamp": datetime.now().isoformat(),
            "source_count": len(documents),
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])
        }

        return {
            "content": documents[latest_idx],
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