# CDT (Content-Dependent Trees)

Ideas for Content-Dependent Trees and other decentralized data structures

## CDTRope Implementation

The CDTRope module implements a content-dependent rope data structure that combines the efficiency of rope operations with content-aware chunking. This experimental implementation uses AVL-style balancing and SHA-256 hashing for maintaining structural integrity.

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
🧬 Initialization
- Empty rope creation
- Hash consistency for empty structure

✂️  Text Operations
- Delete from middle of text
- Delete from beginning
- Delete from end
- Substring retrieval
- Large text operations

🧩 Content-Based Features
- Chunking algorithm consistency
- Hash behavior with modifications
- Node integrity verification

🌳 Tree Operations
- Rebalancing after multiple insertions
- Edge case handling for deletions/insertions
- Mixed operation sequences

🔄 Consistency Checks
- Identical content produces identical chunks
- Hash changes properly track modifications
- Internal node property consistency
```

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

MIT
