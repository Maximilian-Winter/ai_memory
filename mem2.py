import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union
import json


class SemanticMemory:
    def __init__(self, persist_directory: str = "./memory"):
        """Initialize the semantic memory system"""
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create or get existing collections
        try:
            self.client.delete_collection("immediate_memory")
            self.client.delete_collection("working_memory")
            self.client.delete_collection("long_term_memory")
        except:
            pass

        self.immediate = self.client.create_collection(
            name="immediate_memory",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.working = self.client.create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine"}
        )

        self.long_term = self.client.create_collection(
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
        results = []
        seen_contents = set()  # Track unique contents

        # Generate query embedding
        query_embedding = self.encoder.encode(query).tolist()

        # Query each memory layer
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
                    where=context_filter
                )

                # Process results from this layer
                if layer_results['documents']:
                    for doc, meta, dist in zip(
                            layer_results['documents'][0],
                            layer_results['metadatas'][0],
                            layer_results['distances'][0]
                    ):
                        # Only add if content is unique
                        if doc not in seen_contents:
                            seen_contents.add(doc)
                            # Convert distance to similarity score
                            similarity = 1 - float(dist)
                            results.append({
                                'content': doc,
                                'metadata': meta,
                                'similarity': similarity,
                                'memory_type': mem_type
                            })
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")

        # Sort by similarity and limit to requested number
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:n_results]

    def _consolidate_patterns(self):
        """Consolidate patterns across memory layers"""
        immediate_memories = self.immediate.get()

        if not immediate_memories['documents']:
            return

        # Get embeddings
        embeddings = [
            self.encoder.encode(doc)
            for doc in immediate_memories['documents']
        ]

        clusters = self._cluster_embeddings(embeddings)

        # Process each cluster
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < 2:  # Skip singleton clusters
                continue

            pattern = self._extract_pattern(
                [immediate_memories['documents'][i] for i in cluster],
                [immediate_memories['metadatas'][i] for i in cluster],
                [embeddings[i] for i in cluster]
            )

            pattern_id = f"pattern_{datetime.now().isoformat()}_{cluster_idx}"

            # Generate embedding for pattern
            pattern_embedding = self.encoder.encode(pattern['content']).tolist()

            # Store pattern in working memory
            self.working.add(
                documents=[pattern['content']],
                metadatas=[pattern['metadata']],
                embeddings=[pattern_embedding],
                ids=[pattern_id]
            )

        # Cleanup immediate memory (keep last 100)
        if len(immediate_memories['ids']) > 100:
            oldest_ids = immediate_memories['ids'][:-100]
            self.immediate.delete(ids=oldest_ids)

    def _cluster_embeddings(self, embeddings: List[np.ndarray],
                            threshold: float = 0.8) -> List[List[int]]:
        """Cluster embeddings using cosine similarity"""
        # Convert list of embeddings to 2D numpy array
        embeddings_array = np.stack(embeddings)

        # Compute cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings_array / norms[:, np.newaxis]
        similarity_matrix = np.dot(normalized, normalized.T)

        # Find clusters
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
                         metadatas: List[Dict],
                         embeddings: List[np.ndarray]) -> Dict:
        """Extract pattern from cluster of similar memories"""
        # Use most recent document as representative
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

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            "immediate_count": self.immediate.count(),
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count()
        }


if __name__ == "__main__":
    # Initialize memory system
    memory = SemanticMemory("./semantic_memory")

    # Store some memories
    memory.store(
        "The sky was particularly blue today",
        context={"category": "observation"}
    )

    memory.store(
        "I learned about quantum computing",
        context={"category": "learning"}
    )

    # Test recall with different queries
    for query in [
        "What did you learn about?",
        "What was the weather like?",
    ]:
        print(f"\nQuery: {query}")
        results = memory.recall(query, n_results=3)

        for result in results:
            print(f"\nMemory: {result['content']}")
            print(f"Type: {result['memory_type']}")
            print(f"Similarity: {result['similarity']:.3f}")