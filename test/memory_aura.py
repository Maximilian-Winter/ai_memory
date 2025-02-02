import time

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import shutil
import os
from collections import Counter
import threading
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class MemoryStats:
    access_count: int = 0
    last_access: datetime = None
    consolidation_count: int = 0
    pattern_strength: float = 0.0


class SemanticMemory:
    def __init__(self, persist_directory: str = "./memory",
                 batch_size: int = 50,
                 consolidation_interval: int = 300):
        """
        Initialize the semantic memory system with improved configuration

        Args:
            persist_directory: Directory for persistent storage
            batch_size: Size of batches for operations
            consolidation_interval: Seconds between consolidation runs
        """
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.batch_size = batch_size
        self.consolidation_interval = consolidation_interval

        # Enhanced memory collections
        self.immediate = self.client.get_or_create_collection(
            name="immediate_memory",
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
        )

        self.working = self.client.get_or_create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
        )

        self.long_term = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
        )

        # Memory statistics tracking
        self.memory_stats = {}
        self.lock = threading.Lock()

        # Start background consolidation
        self._start_background_consolidation()

    def _start_background_consolidation(self):
        """Start background thread for memory consolidation"""

        def consolidation_loop():
            while True:
                try:
                    self._consolidate_patterns()
                except Exception as e:
                    logging.error(f"Consolidation error: {str(e)}")
                time.sleep(self.consolidation_interval)

        thread = threading.Thread(target=consolidation_loop, daemon=True)
        thread.start()

    def store(self, content: str, context: Optional[Dict] = None,
              batch: bool = True) -> str:
        """
        Store new memory with improved batching and error handling
        """
        timestamp = datetime.now().isoformat()
        memory_id = f"mem_{timestamp}"

        try:
            metadata = {
                "timestamp": timestamp,
                "type": "memory",
                "version": "2.0"
            }
            if context:
                metadata.update(context)

            embedding = self._get_embedding(content)

            if batch:
                # Add to batch queue
                self._add_to_batch(content, metadata, embedding, memory_id)
            else:
                # Immediate storage
                self._store_single(content, metadata, embedding, memory_id)

            # Initialize stats
            with self.lock:
                self.memory_stats[memory_id] = MemoryStats(
                    access_count=0,
                    last_access=datetime.now(),
                    consolidation_count=0,
                    pattern_strength=1.0
                )

            return memory_id

        except Exception as e:
            logging.error(f"Storage error: {str(e)}")
            raise

    def recall(self, query: str, n_results: int = 5,
               context_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Enhanced recall with reinforcement learning
        """
        query_embedding = self._get_embedding(query)
        results = []
        seen_contents = set()

        # Query each memory layer with error handling
        for collection, mem_type, weight in [
            (self.immediate, "immediate", 1.0),
            (self.working, "working", 0.8),
            (self.long_term, "long_term", 0.6)
        ]:
            try:
                if collection.count() == 0:
                    continue

                layer_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results * 2, collection.count()),  # Over-fetch for deduplication
                    where=context_filter if context_filter else None
                )

                # Process and weight results
                processed_results = self._process_layer_results(
                    layer_results, mem_type, weight)
                results.extend(processed_results)

                # Update access statistics
                self._update_access_stats(layer_results['ids'][0])

            except Exception as e:
                logging.error(f"Recall error in {mem_type}: {str(e)}")
                continue

        # Deduplicate and sort results
        return self._deduplicate_and_sort_results(results, n_results)

    def _get_embedding(self, content: str) -> List[float]:
        """Generate embedding with error handling"""
        try:
            return self.encoder.encode(content).tolist()
        except Exception as e:
            logging.error(f"Embedding error: {str(e)}")
            raise

    def _consolidate_patterns(self):
        """
        Enhanced pattern consolidation with adaptive thresholds
        """
        try:
            immediate_memories = self.immediate.get()
            if not immediate_memories['documents']:
                return

            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(immediate_memories['documents']), self.batch_size):
                batch = immediate_memories['documents'][i:i + self.batch_size]
                batch_embeddings = [self._get_embedding(doc) for doc in batch]
                embeddings.extend(batch_embeddings)

            # Adaptive clustering
            clusters = self._adaptive_clustering(embeddings)

            # Process clusters in parallel
            with ThreadPoolExecutor() as executor:
                futures = []
                for cluster_idx, cluster in enumerate(clusters):
                    if len(cluster) < 2:  # Skip singleton clusters
                        continue

                    futures.append(executor.submit(
                        self._process_cluster,
                        cluster,
                        immediate_memories,
                        cluster_idx
                    ))

                # Wait for all cluster processing to complete
                for future in futures:
                    future.result()

            # Cleanup immediate memory
            self._cleanup_immediate_memory(immediate_memories)

        except Exception as e:
            logging.error(f"Consolidation error: {str(e)}")
            raise

    def _adaptive_clustering(self, embeddings: List[np.ndarray]) -> List[List[int]]:
        """
        Clustering with adaptive threshold based on density
        """
        if not embeddings:
            return []

        embeddings_array = np.stack(embeddings)

        # Compute pairwise similarities
        norms = np.linalg.norm(embeddings_array, axis=1)
        norms[norms == 0] = 1e-10  # Prevent division by zero
        normalized = embeddings_array / norms[:, np.newaxis]
        similarity_matrix = np.dot(normalized, normalized.T)

        # Compute adaptive threshold based on similarity distribution
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        threshold = np.percentile(similarities, 75)  # Use 75th percentile as threshold

        # Clustering with adaptive threshold
        clusters = []
        used_indices = set()

        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            # Find all points within threshold
            similar_points = np.where(similarity_matrix[i] > threshold)[0]

            # Create cluster if enough similar points
            if len(similar_points) >= 2:
                cluster = similar_points.tolist()
                clusters.append(cluster)
                used_indices.update(cluster)
            else:
                clusters.append([i])
                used_indices.add(i)

        return clusters

    def _process_cluster(self, cluster: List[int],
                         memories: Dict,
                         cluster_idx: int) -> None:
        """
        Process individual cluster for pattern formation
        """
        try:
            pattern = self._extract_pattern(
                [memories['documents'][i] for i in cluster],
                [memories['metadatas'][i] for i in cluster]
            )

            pattern_embedding = self._get_embedding(pattern['content'])
            pattern_id = f"pattern_{datetime.now().isoformat()}_{cluster_idx}"

            # Check for similar existing patterns
            existing = self.working.query(
                query_embeddings=[pattern_embedding],
                n_results=1
            )

            if existing['distances'] and existing['distances'][0]:
                if existing['distances'][0][0] <= 0.1:  # Very similar pattern exists
                    self._reinforce_pattern(existing['ids'][0])
                    return

            # Store new pattern
            self.working.add(
                documents=[pattern['content']],
                metadatas=[pattern['metadata']],
                embeddings=[pattern_embedding],
                ids=[pattern_id]
            )

        except Exception as e:
            logging.error(f"Cluster processing error: {str(e)}")
            raise

    def _reinforce_pattern(self, pattern_id: str):
        """
        Reinforce existing pattern
        """
        with self.lock:
            if pattern_id in self.memory_stats:
                stats = self.memory_stats[pattern_id]
                stats.consolidation_count += 1
                stats.pattern_strength *= 1.1  # Increase pattern strength

    def _cleanup_immediate_memory(self, memories: Dict):
        """
        Cleanup immediate memory with sophisticated priority
        """
        if len(memories['ids']) > 100:
            # Sort by access patterns and age
            priorities = []
            for idx, mem_id in enumerate(memories['ids']):
                stats = self.memory_stats.get(mem_id, MemoryStats())
                age = (datetime.now() - stats.last_access).total_seconds() if stats.last_access else float('inf')
                priority = stats.access_count / (age + 1)  # Higher priority for frequently accessed recent memories
                priorities.append((idx, priority))

            # Keep top memories
            priorities.sort(key=lambda x: x[1])
            to_remove = [memories['ids'][idx] for idx, _ in priorities[:-100]]
            self.immediate.delete(ids=to_remove)

    def get_stats(self) -> Dict:
        """
        Get enhanced memory system statistics
        """
        with self.lock:
            stats = {
                "immediate_count": self.immediate.count(),
                "working_count": self.working.count(),
                "long_term_count": self.long_term.count(),
                "total_access_count": sum(stat.access_count for stat in self.memory_stats.values()),
                "average_pattern_strength": np.mean(
                    [stat.pattern_strength for stat in self.memory_stats.values()]) if self.memory_stats else 0
            }
        return stats