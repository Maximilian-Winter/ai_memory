import shutil
import os
from datetime import datetime, timedelta
import time
import numpy as np
from enhanced_semantic_memory import EnhancedSemanticMemory, MemoryConfig

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def test_memory_system():
    # Clean up any existing test database
    persist_directory = "./test_enhanced_memory"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # Initialize with custom configuration
    config = MemoryConfig(
        immediate_capacity=100,
        working_memory_threshold=0.8,
        long_term_threshold=0.9,
        min_pattern_support=2,
        consolidation_window=3600,  # 1 hour for testing
        attention_decay=0.1
    )

    memory = EnhancedSemanticMemory(persist_directory, config)

    print_separator("Initial System Test")
    
    # Test 1: Basic Storage and Recall
    print("\nTest 1: Basic Storage and Recall")
    
    # Store a series of related memories about quantum computing
    quantum_memories = [
        "Quantum computing uses qubits which can exist in superposition",
        "Unlike classical bits, qubits can represent multiple states simultaneously",
        "Superposition is a fundamental principle of quantum mechanics",
        "Quantum entanglement allows qubits to be correlated",
        "Quantum algorithms can solve certain problems exponentially faster"
    ]
    
    for memory_text in quantum_memories:
        memory.store(
            memory_text,
            context={"category": "quantum_computing", "importance": "high"}
        )

    # Test immediate recall
    print("\nRecalling quantum computing information:")
    results = memory.recall("Tell me about quantum computing", n_results=3)
    for r in results:
        print(f"\nContent: {r['content']}")
        print(f"Memory Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")
        print(f"Metadata: {r['metadata']}")

    # Test 2: Pattern Formation
    print_separator("Pattern Formation Test")
    
    # Add similar memories to trigger pattern formation
    pattern_memories = [
        "Quantum superposition allows multiple states simultaneously",
        "In quantum systems, particles can exist in multiple states",
        "Superposition is key to quantum computing's power",
        "The principle of superposition distinguishes quantum from classical"
    ]
    
    for memory_text in pattern_memories:
        memory.store(
            memory_text,
            context={"category": "quantum_computing", "subtopic": "superposition"}
        )

    print("\nChecking for patterns about superposition:")
    results = memory.recall("What is quantum superposition?", n_results=3)
    for r in results:
        print(f"\nContent: {r['content']}")
        print(f"Memory Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")
        print(f"Metadata: {r['metadata']}")

    # Test 3: Context Filtering
    print_separator("Context Filtering Test")
    
    # Add memories with different contexts
    memory.store(
        "Classical computers use bits as fundamental units",
        context={"category": "classical_computing", "importance": "high"}
    )
    
    memory.store(
        "Neural networks are inspired by biological brains",
        context={"category": "machine_learning", "importance": "high"}
    )

    print("\nRecalling only quantum computing memories:")
    results = memory.recall(
        "Tell me about computing",
        n_results=3,
        context_filter={"category": "quantum_computing"}
    )
    for r in results:
        print(f"\nContent: {r['content']}")
        print(f"Memory Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")

    # Test 4: Attention Mechanism
    print_separator("Attention Mechanism Test")
    
    # Repeatedly query certain memories to increase their attention weights
    for _ in range(3):
        memory.recall("What is superposition?", n_results=2)
        time.sleep(0.1)  # Small delay to simulate time passing

    print("\nChecking attention-weighted recall:")
    results = memory.recall("Tell me about quantum concepts", n_results=3)
    for r in results:
        print(f"\nContent: {r['content']}")
        print(f"Memory Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")

    # Test 5: Information Content and Pattern Quality
    print_separator("Information Content Test")
    
    # Add memories with varying information content
    memory.store(
        "Quantum error correction is essential for building reliable quantum computers",
        context={"category": "quantum_computing", "subtopic": "error_correction"}
    )
    
    memory.store(
        "Error correction helps maintain quantum states",
        context={"category": "quantum_computing", "subtopic": "error_correction"}
    )

    # Test 6: System Statistics
    print_separator("System Statistics")
    
    stats = memory.get_stats()
    print("\nCurrent system statistics:")
    print(f"Immediate memories: {stats['immediate_count']}")
    print(f"Working memory patterns: {stats['working_count']}")
    print(f"Long-term memories: {stats['long_term_count']}")
    print(f"Access pattern entropy: {stats['access_pattern_entropy']:.3f}")
    print(f"Pattern quality: {stats['pattern_quality']:.3f}")
    
    if "performance_trends" in stats:
        print("\nPerformance trends:")
        print(f"Pattern quality trend: {stats['performance_trends']['pattern_quality_trend']}")
        print(f"Access entropy trend: {stats['performance_trends']['access_entropy_trend']}")

    # Test 7: Mixed Recall Test
    print_separator("Mixed Recall Test")
    
    # Add some unrelated memories
    memory.store(
        "The weather is beautiful today",
        context={"category": "weather", "importance": "low"}
    )
    
    memory.store(
        "Coffee helps with productivity",
        context={"category": "productivity", "importance": "medium"}
    )

    print("\nTesting general recall with mixed memories:")
    results = memory.recall("Tell me what you remember", n_results=5)
    for r in results:
        print(f"\nContent: {r['content']}")
        print(f"Memory Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")
        print(f"Category: {r['metadata'].get('category', 'unknown')}")

    # Test 8: Temporal Dynamics
    print_separator("Temporal Dynamics Test")
    
    print("\nWaiting for consolidation window...")
    time.sleep(2)  # Simulate time passing
    
    # Add new related memories
    memory.store(
        "Quantum entanglement enables quantum teleportation",
        context={"category": "quantum_computing", "subtopic": "entanglement"}
    )
    
    print("\nChecking temporal effects on recall:")
    results = memory.recall("What do you know about quantum phenomena?", n_results=3)
    for r in results:
        print(f"\nContent: {r['content']}")
        print(f"Memory Type: {r['memory_type']}")
        print(f"Similarity: {r['similarity']:.3f}")
        print(f"Timestamp: {r['metadata']['timestamp']}")

    # Final System Statistics
    print_separator("Final System State")
    
    final_stats = memory.get_stats()
    print("\nFinal system statistics:")
    print(json.dumps(final_stats, indent=2))

if __name__ == "__main__":
    test_memory_system()