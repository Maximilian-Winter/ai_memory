import chromadb
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import shutil
import os
from dataclasses import dataclass
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import faiss
from collections import Counter

@dataclass
class MemoryConfig:
    """Configuration for memory system parameters"""
    immediate_capacity: int = 1000
    working_memory_threshold: float = 0.8
    long_term_threshold: float = 0.9
    min_pattern_support: int = 3
    consolidation_window: int = 86400  # 24 hours in seconds
    attention_decay: float = 0.1
    embedding_dim: int = 384  # For all-MiniLM-L6-v2

class EnhancedSemanticMemory:
    def __init__(self, persist_directory: str = "./memory", config: Optional[MemoryConfig] = None):
        """Initialize the enhanced semantic memory system"""
        self.config = config or MemoryConfig()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.config.embedding_dim)
        
        # Initialize memory layers
        self.immediate = self._create_memory_collection("immediate_memory")
        self.working = self._create_memory_collection("working_memory")
        self.long_term = self._create_memory_collection("long_term_memory")
        
        # Initialize attention mechanism
        self.attention_weights = {}
        self.access_counts = Counter()
        
        # Meta-learning state
        self.performance_history = []
        self.parameter_history = []

    def _create_memory_collection(self, name: str):
        """Create a memory collection with enhanced metadata"""
        return self.client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100
            }
        )

    def store(self, content: str, context: Optional[Dict] = None) -> str:
        """Store new memory with enhanced context and attention"""
        timestamp = datetime.now().isoformat()
        
        # Compute information-theoretic measures
        embedding = self.encoder.encode(content).tolist()
        info_content = self._compute_information_content(embedding)
        
        metadata = {
            "timestamp": timestamp,
            "type": "memory",
            "info_content": info_content,
            "attention_weight": 1.0
        }
        if context:
            metadata.update(context)

        memory_id = f"mem_{timestamp}"
        
        # Store in immediate memory
        self.immediate.add(
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[memory_id]
        )
        
        # Update FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Trigger enhanced consolidation
        self._consolidate_patterns()
        return memory_id

    def recall(self, query: str, n_results: int = 5, 
              context_filter: Optional[Dict] = None) -> List[Dict]:
        """Enhanced recall with attention mechanism"""
        query_embedding = self.encoder.encode(query).tolist()
        results = []
        seen_contents = set()

        # Update attention weights
        self._update_attention_weights()

        for collection, mem_type in [
            (self.immediate, "immediate"),
            (self.working, "working"),
            (self.long_term, "long_term")
        ]:
            try:
                if collection.count() == 0:
                    continue

                # Apply attention-weighted search
                layer_results = self._attention_weighted_search(
                    collection, query_embedding, n_results, context_filter
                )
                results.extend(self._format_results(layer_results, mem_type))
                
                # Update access statistics
                for result in layer_results['ids'][0]:
                    self.access_counts[result] += 1
                
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")

        # Apply temporal dynamics and attention for ranking
        results = self._apply_temporal_attention(results)
        
        # Deduplicate and sort
        unique_results = self._deduplicate_results(results)
        return unique_results[:n_results]

    def _compute_information_content(self, embedding: List[float]) -> float:
        """Compute information content using entropy"""
        # Normalize embedding to probability distribution
        prob_dist = np.abs(embedding) / np.sum(np.abs(embedding))
        return float(entropy(prob_dist))

    def _update_attention_weights(self):
        """Update attention weights based on access patterns"""
        current_time = datetime.now().timestamp()
        
        for memory_id, count in self.access_counts.items():
            if memory_id not in self.attention_weights:
                self.attention_weights[memory_id] = 1.0
            
            # Apply temporal decay and access frequency boost
            time_decay = np.exp(-self.config.attention_decay * 
                              (current_time - self._get_memory_timestamp(memory_id)))
            frequency_boost = np.log1p(count)
            
            self.attention_weights[memory_id] *= time_decay * (1 + frequency_boost)

    def _attention_weighted_search(self, collection, query_embedding: List[float],
                                 n_results: int, context_filter: Optional[Dict]) -> Dict:
        """Perform attention-weighted similarity search"""
        # Get base results
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * 2, collection.count()),  # Get extra results for reranking
            where=context_filter if context_filter and len(context_filter) > 0 else None
        )
        
        if not results['ids'][0]:
            return results
            
        # Apply attention weights
        weighted_distances = []
        for i, memory_id in enumerate(results['ids'][0]):
            base_distance = results['distances'][0][i]
            attention_weight = self.attention_weights.get(memory_id, 1.0)
            weighted_distances.append(base_distance * (1 / attention_weight))
            
        # Rerank results
        ranked_indices = np.argsort(weighted_distances)[:n_results]
        
        # Reconstruct results
        return {
            'ids': [[results['ids'][0][i] for i in ranked_indices]],
            'distances': [[weighted_distances[i] for i in ranked_indices]],
            'metadatas': [[results['metadatas'][0][i] for i in ranked_indices]],
            'documents': [[results['documents'][0][i] for i in ranked_indices]]
        }

    def _consolidate_patterns(self):
        """Enhanced pattern consolidation with hierarchical clustering"""
        immediate_memories = self.immediate.get()
        
        if not immediate_memories['documents']:
            return
            
        # Generate embeddings
        embeddings = np.array([
            self.encoder.encode(doc)
            for doc in immediate_memories['documents']
        ])
        
        # Perform hierarchical clustering
        clusters = self._hierarchical_clustering(embeddings)
        
        # Process patterns across hierarchy levels
        for level, level_clusters in enumerate(clusters):
            self._process_level_patterns(
                level_clusters,
                immediate_memories,
                level
            )
            
        # Update meta-learning state
        self._update_meta_learning_state()
        
        # Cleanup immediate memory
        self._cleanup_immediate_memory(immediate_memories)

    def _hierarchical_clustering(self, embeddings: np.ndarray) -> List[List[List[int]]]:
        """Perform hierarchical clustering at multiple thresholds"""
        thresholds = [0.7, 0.8, 0.9]  # Multiple levels of granularity
        clusters_hierarchy = []
        
        for threshold in thresholds:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage='complete'
            ).fit(embeddings)
            
            # Group indices by cluster
            clusters = [[] for _ in range(max(clustering.labels_) + 1)]
            for idx, label in enumerate(clustering.labels_):
                clusters[label].append(idx)
                
            clusters_hierarchy.append(clusters)
            
        return clusters_hierarchy

    def _process_level_patterns(self, clusters: List[List[int]], 
                              memories: Dict, level: int):
        """Process patterns at a specific hierarchy level"""
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < self.config.min_pattern_support:
                continue
                
            pattern = self._extract_enhanced_pattern(
                [memories['documents'][i] for i in cluster],
                [memories['metadatas'][i] for i in cluster],
                level
            )
            
            pattern_id = f"pattern_L{level}_{datetime.now().isoformat()}_{cluster_idx}"
            pattern_embedding = self.encoder.encode(pattern['content']).tolist()
            
            # Store pattern in working memory if novel
            if not self._pattern_exists(pattern_embedding):
                self.working.add(
                    documents=[pattern['content']],
                    metadatas=[pattern['metadata']],
                    embeddings=[pattern_embedding],
                    ids=[pattern_id]
                )

    def _extract_enhanced_pattern(self, documents: List[str],
                                metadatas: List[Dict],
                                level: int) -> Dict:
        """Extract enhanced pattern with temporal and information-theoretic features"""
        # Find most informative example
        info_contents = [meta.get('info_content', 0) for meta in metadatas]
        informative_idx = max(range(len(info_contents)), 
                            key=lambda i: info_contents[i])
        
        pattern_metadata = {
            "type": "pattern",
            "timestamp": datetime.now().isoformat(),
            "source_count": len(documents),
            "hierarchy_level": level,
            "info_content": np.mean(info_contents),
            "temporal_span": self._compute_temporal_span(metadatas),
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])
        }
        
        return {
            "content": documents[informative_idx],
            "metadata": pattern_metadata
        }

    def _compute_temporal_span(self, metadatas: List[Dict]) -> float:
        """Compute temporal span of a pattern"""
        timestamps = [
            datetime.fromisoformat(meta['timestamp']).timestamp()
            for meta in metadatas
        ]
        return max(timestamps) - min(timestamps)

    def _update_meta_learning_state(self):
        """Update meta-learning parameters based on system performance"""
        # Record current performance metrics
        current_performance = {
            "immediate_count": self.immediate.count(),
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count(),
            "access_pattern_entropy": self._compute_access_pattern_entropy(),
            "pattern_quality": self._evaluate_pattern_quality()
        }
        
        self.performance_history.append(current_performance)
        
        # Adjust parameters if needed
        if len(self.performance_history) >= 5:
            self._adjust_parameters()

    def _compute_access_pattern_entropy(self) -> float:
        """Compute entropy of memory access patterns"""
        if not self.access_counts:
            return 0.0
            
        counts = np.array(list(self.access_counts.values()))
        probs = counts / counts.sum()
        return float(entropy(probs))

    def _evaluate_pattern_quality(self) -> float:
        """Evaluate quality of extracted patterns"""
        working_memories = self.working.get()
        if not working_memories['documents']:
            return 0.0
            
        # Compute average information content and support
        info_contents = [
            meta.get('info_content', 0) 
            for meta in working_memories['metadatas']
        ]
        supports = [
            meta.get('source_count', 0) 
            for meta in working_memories['metadatas']
        ]
        
        return float(np.mean(info_contents) * np.mean(supports))

    def _adjust_parameters(self):
        """Adjust system parameters based on performance history"""
        recent_performance = self.performance_history[-5:]
        
        # Adjust consolidation thresholds based on pattern quality trend
        quality_trend = [perf['pattern_quality'] for perf in recent_performance]
        if np.mean(quality_trend) < np.mean(quality_trend[:3]):
            self.config.working_memory_threshold *= 0.95
            self.config.long_term_threshold *= 0.95
        else:
            self.config.working_memory_threshold = min(0.95, 
                self.config.working_memory_threshold * 1.05)
            self.config.long_term_threshold = min(0.98, 
                self.config.long_term_threshold * 1.05)
        
        # Adjust attention decay based on access patterns
        entropy_trend = [perf['access_pattern_entropy'] 
                        for perf in recent_performance]
        if np.mean(entropy_trend) > np.mean(entropy_trend[:3]):
            self.config.attention_decay *= 0.9
        else:
            self.config.attention_decay *= 1.1

    def get_stats(self) -> Dict:
        """Get enhanced memory system statistics"""
        stats = {
            "immediate_count": self.immediate.count(),
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count(),
            "access_pattern_entropy": self._compute_access_pattern_entropy(),
            "pattern_quality": self._evaluate_pattern_quality(),
            "current_parameters": {
                "working_memory_threshold": self.config.working_memory_threshold,
                "long_term_threshold": self.config.long_term_threshold,
                "attention_decay": self.config.attention_decay
            }
        }
        
        # Add performance history summary if available
        if self.performance_history:
            stats["performance_trends"] = {
                "pattern_quality_trend": [
                    p['pattern_quality'] for p in self.performance_history[-5:]
                ],
                "access_entropy_trend": [
                    p['access_pattern_entropy'] 
                    for p in self.performance_history[-5:]
                ]
            }
            
        return stats