# CDT (Content-Dependent Trees)

Ideas for Content-Dependent Trees and other decentralized data structures

## CDTRope Implementation

The CDTRope module implements a content-dependent rope data structure that combines the efficiency of rope operations with content-aware chunking. This experimental implementation uses AVL-style balancing and SHA-256 hashing for maintaining structural integrity.

### Benefits of Rope Data Structures

Rope data structures offer significant advantages over traditional string representations, especially for large texts and frequent modifications:

#### 1. Efficient Memory Management
- Traditional strings require contiguous memory blocks
- Ropes use tree-like structures, allowing fragmented memory allocation
- Minimal data copying during insertions, deletions, and modifications
- Ideal for large documents and memory-constrained environments

#### 2. Performance Optimization
- Most operations (insert, delete, substring) have O(log n) time complexity
- Reduced memory allocation overhead
- Efficient handling of large text documents
- Minimal data movement during text modifications

#### 3. Content-Aware Operations
- Supports intelligent chunking based on content
- Natural split points can be identified using rolling hash algorithms
- Enables more efficient text processing and storage

#### 4. Practical Use Cases
- Text editors handling large files
- Version control systems
- Log file management
- Collaborative editing platforms
- Memory-efficient text processing applications

#### Comparative Performance Example
Consider a 1MB document:
- Traditional String: Inserting a character requires copying ~1MB of data
- CDTRope: Modify just a few tree nodes, minimal data movement

#### Memory Efficiency Illustration
```
Traditional String:    [1MB Contiguous Block]
CDTRope:               Tree Structure
                       /            \
               [256KB Chunk]    [768KB Chunk]
                   /    \           /    \
           [128KB]  [128KB]   [384KB]  [384KB]
```

This flexible structure allows for more intelligent and efficient text handling across various applications.


### Key Features

- Content-aware chunking using rolling hash algorithm
- Consistent hashing for structural verification
- AVL-style tree balancing for performance
- Efficient string operations (insert, delete, substring)
- Memory-efficient representation of large texts

### Implementation Details

- Chunk size: 8 bytes (configurable, production recommended: 64KB)
- Rebalancing threshold: 1.5
- Rolling hash window size: 4 bytes
- Hash function: SHA-256 (truncated to 16 chars)

### API

```python
class CDTRope:
    def __init__(self, text: str = "")
    def insert(self, index: int, text: str) -> None
    def delete(self, start: int, length: int) -> None
    def substring(self, start: int, length: int) -> str
    def get_text(self) -> str
    def get_hash(self) -> str
```

### Test Results

✅ All tests passing:

```
🔄 Tests that identical content produces identical chunk patterns
🧩 Tests the content-based chunking algorithm's consistency
🗑️  Tests deleting text from the beginning of the rope
✂️  Tests deleting text from the end of the rope
✂️  Tests deleting text from the middle of the rope
🎯 Tests edge cases in delete operations like negative indices and beyond text length
🎯 Tests edge cases in insert operations like negative indices and beyond text length
🧬 Tests initialization of empty rope data structure
🔐 Tests hash consistency and changes across text modifications
⛓️ Tests that hash reflects the actual tree structure while maintaining consistency
🔄 Tests hash equivalence after series of operations that should result in same content
🔐 Tests hash behavior and content consistency through split and merge operations
🔍 Tests hash sensitivity to content order and structure
🌳 Tests hash stability through tree rebalancing operations
📚 Tests handling of large text operations including insertion and deletion
🏗️  Tests internal node properties like height, weight, and hash consistency
🌳 Tests if the tree maintains balance after multiple insertions at the beginning
📝 Tests a sequence of mixed operations (insert/delete) on the rope
📄 Tests retrieving substrings from the rope
```

### Chunk Size Considerations

The choice of chunk size in production environments is critical and depends on several factors. Here's a comprehensive guide:

#### Real-world References
- Git: Variable chunks averaging ~8KB
- ZFS: 128KB default block size
- Btrfs: 4KB-64KB blocks
- Dropbox: Historically used 4MB chunks

#### Default Production Setting
```python
CHUNK_SIZE = 64 * 1024  # 64KB is a good starting point
```

#### Performance Impact

Small Chunks (< 4KB):
- Higher metadata overhead
- Increased fragmentation
- Higher hash processing cost

Large Chunks (> 1MB):
- Reduced deduplication efficiency
- Higher memory usage
- More expensive rehashing for small changes

#### Memory Usage Considerations
Each node requires memory for:
- Pointers (16-24 bytes)
- Hash (32 bytes typically)
- Metadata (~20-40 bytes)

Example: With 64KB chunks, 1GB of data would require approximately 16K nodes

#### Recommended Sizes by Use Case
- Small files: 4KB-16KB
- Documents/code: 16KB-64KB
- Media/binaries: 64KB-256KB
- Streaming: 256KB-1MB

### Performance Considerations

- The implementation uses content-based chunking to find natural split points in text
- AVL-style balancing ensures O(log n) complexity for most operations
- Rolling hash window helps identify chunk boundaries efficiently
- Tree rebalancing threshold of 1.5 maintains balance while reducing rebalancing frequency

### Usage Example

```python
# Create a new rope
rope = CDTRope("Hello World")

# Insert text
rope.insert(5, " Beautiful")  # "Hello Beautiful World"

# Delete text
rope.delete(0, 6)  # "Beautiful World"

# Get substring
substr = rope.substring(0, 9)  # "Beautiful"

# Get structural hash
hash = rope.get_hash()
```

### Future Improvements

1. Implement concurrent modification support
2. Add merge operation with conflict detection
3. Optimize memory usage for very large texts
4. Add versioning support
5. Implement efficient serialization/deserialization

### Contributing

This is an experimental implementation. Contributions and suggestions are welcome to improve the functionality and efficiency of the content-dependent tree data structures.

## License

MIT License.
