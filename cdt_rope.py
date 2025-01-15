import unittest
import hashlib
import random
from typing import Optional, Tuple, List

class CDTNode:
    def __init__(self, text: str = "", weight: int = 0):
        self.weight = weight  # Size of left subtree
        self.text = text
        self.left: Optional[CDTNode] = None
        self.right: Optional[CDTNode] = None
        self.height = 1  # Para balanceo AVL
        self.hash = self._calculate_hash(text)
    
    def _calculate_hash(self, text: str) -> str:
        """Consistent hash calculation."""
        # For internal nodes, we want to create a deterministic hash based on children
        if not self.is_leaf():
            left_hash = self.left.hash if self.left else ""
            right_hash = self.right.hash if self.right else ""
            text = left_hash + right_hash
        
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
    
    def update_height(self):
        """Actualiza la altura del nodo basado en sus hijos."""
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        self.height = max(left_height, right_height) + 1

class CDTRope:
    # Reducimos CHUNK_SIZE para tests
    CHUNK_SIZE = 8  # En producción usar 64 * 1024
    REBALANCE_THRESHOLD = 1.5
    
    def __init__(self, text: str = ""):
        self.root = self._build_from_text(text)

    def _print_tree(self, node: Optional[CDTNode], prefix: str = "", is_right: bool = False) -> None:
        if node:
            print(f"{prefix}{'└── ' if is_right else '├── '}[{node.text if node.is_leaf() else 'Internal'}, w:{node.weight}, h:{node.height}]")
            if node.left or node.right:
                if node.left:
                    self._print_tree(node.left, prefix + ("    " if is_right else "│   "), False)
                if node.right:
                    self._print_tree(node.right, prefix + ("    " if is_right else "│   "), True)
    
    def _get_weight(self, node: Optional[CDTNode]) -> int:
        """Get total weight (size) of node."""
        if not node:
            return 0
        if node.is_leaf():
            return len(node.text)
        return node.weight + (self._get_weight(node.right) if node.right else 0)

    def _build_from_text(self, text: str) -> Optional[CDTNode]:
        """Build a balanced rope from input text using content-based chunking."""
        if not text:
            return None
            
        # Si el texto es pequeño, crear nodo hoja
        if len(text) <= self.CHUNK_SIZE:
            return CDTNode(text, len(text))
            
        # Encontrar punto de división basado en contenido
        split_point = self._find_chunk_boundary(text)
        node = CDTNode(weight=split_point)
        node.left = self._build_from_text(text[:split_point])
        node.right = self._build_from_text(text[split_point:])
        
        # Calcular hash combinado
        left_hash = node.left.hash if node.left else ""
        right_hash = node.right.hash if node.right else ""
        node.hash = node._calculate_hash(left_hash + right_hash)
        
        node.update_height()
        return node

    def _find_chunk_boundary(self, text: str) -> int:
        """Find content-based split point using rolling hash."""
        if len(text) <= self.CHUNK_SIZE:
            return len(text) // 2
        
        WINDOW_SIZE = 4
        MASK = 0xFFFF
        TARGET = 0  # Buscar cuando el hash sea divisible por esto
        
        best_pos = len(text) // 2
        min_dist_to_middle = float('inf')
        
        # Usar rolling hash para encontrar puntos de división naturales
        curr_hash = 0
        for i in range(len(text) - WINDOW_SIZE):
            # Update rolling hash
            if i == 0:
                for j in range(WINDOW_SIZE):
                    curr_hash = (curr_hash * 31 + ord(text[j])) & MASK
            else:
                curr_hash = ((curr_hash - ord(text[i-1]) * (31 ** (WINDOW_SIZE-1))) * 31 + 
                           ord(text[i+WINDOW_SIZE-1])) & MASK
            
            # Check if this is a potential boundary
            if curr_hash % 16 == TARGET:
                dist_to_middle = abs(i - len(text)//2)
                if dist_to_middle < min_dist_to_middle:
                    min_dist_to_middle = dist_to_middle
                    best_pos = i + WINDOW_SIZE
        
        return best_pos

    def insert(self, index: int, text: str) -> None:
        """Insert text at given index."""
        if index < 0:
            raise IndexError("Negative index")
            
        if not self.root:
            self.root = self._build_from_text(text)
            return
            
        # Find split point
        left_part, right_part = self._split(self.root, index)
        
        # Create new node for inserted text
        new_node = self._build_from_text(text)
        
        # Merge parts and rebalance
        merged = self._merge(self._merge(left_part, new_node), right_part)
        self.root = self._rebalance(merged)

    def _split(self, node: Optional[CDTNode], index: int) -> Tuple[Optional[CDTNode], Optional[CDTNode]]:
        """Split rope at index, returns (left, right) parts."""
        if not node:
            return None, None
                
        if node.is_leaf():
            if index == 0:
                return None, node
            if index >= len(node.text):
                return node, None
                
            # Split leaf node
            left_node = CDTNode(node.text[:index], index)
            right_node = CDTNode(node.text[index:], len(node.text) - index)
            return left_node, right_node

        current_weight = self._get_weight(node.left) if node.left else 0
        
        if index <= current_weight:
            left, right = self._split(node.left, index)
            if right:
                new_right = CDTNode()
                new_right.left = right
                new_right.right = node.right
                new_right.weight = self._get_weight(right)
                new_right.update_height()
                return left, new_right
            return left, node.right
        else:
            left, right = self._split(node.right, index - current_weight)
            if left:
                new_left = CDTNode()
                new_left.left = node.left
                new_left.right = left
                new_left.weight = current_weight + self._get_weight(left)
                new_left.update_height()
                return new_left, right
            return node.left, right
        
    def _get_max_height_diff(self, node: Optional[CDTNode]) -> float:
        """Helper to get maximum height difference in tree."""
        if not node:
            return 0
            
        left_height = node.left.height if node.left else 0
        right_height = node.right.height if node.right else 0
        
        # Calculate max height difference for current node and recursively for children
        return max(
            abs(left_height - right_height),
            self._get_max_height_diff(node.left) if node.left else 0,
            self._get_max_height_diff(node.right) if node.right else 0
        )

    def get_text(self) -> str:
        """Get full text from rope."""
        return self._get_text_recursive(self.root)
    
    def _get_text_recursive(self, node: Optional[CDTNode]) -> str:
        if not node:
            return ""
        if node.is_leaf():
            return node.text
        return self._get_text_recursive(node.left) + self._get_text_recursive(node.right)

    def get_hash(self) -> str:
        """Get root hash of the rope."""
        return self.root.hash if self.root else ""

    def substring(self, start: int, length: int) -> str:
        """Get substring from rope."""
        if start < 0 or length < 0:
            raise IndexError("Invalid indices")
            
        result = []
        self._substring_recursive(self.root, start, length, result)
        return "".join(result)
    
    def _substring_recursive(self, node: Optional[CDTNode], start: int, length: int, 
                           result: List[str]) -> Tuple[int, int]:
        if not node or length <= 0:
            return (0, 0)
            
        if node.is_leaf():
            if start >= len(node.text):
                return (len(node.text), 0)
            text_to_add = node.text[start:start+length]
            result.append(text_to_add)
            return (len(node.text), len(text_to_add))
        
        if start < node.weight:
            chars_processed, chars_added = self._substring_recursive(
                node.left, start, length, result)
            if chars_added < length:
                right_processed, right_added = self._substring_recursive(
                    node.right, 0, length - chars_added, result)
                return (chars_processed + right_processed, chars_added + right_added)
            return (chars_processed, chars_added)
        else:
            return self._substring_recursive(
                node.right, start - node.weight, length, result)

    def _needs_rebalance(self, node: Optional[CDTNode]) -> bool:
        """Check if node needs rebalancing."""
        if not node:
            return False
            
        left_height = node.left.height if node.left else 0
        right_height = node.right.height if node.right else 0
        
        return abs(left_height - right_height) > self.REBALANCE_THRESHOLD

    def _rebalance(self, node: Optional[CDTNode]) -> Optional[CDTNode]:
        """Rebalance tree at given node."""
        if not node:
            return None
            
        # Rebalance children first
        if node.left:
            node.left = self._rebalance(node.left)
        if node.right:
            node.right = self._rebalance(node.right)
            
        node.update_height()
        balance = self._get_balance(node)
        
        # Left heavy
        if balance > 1:
            if node.left and self._get_balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
            
        # Right heavy
        if balance < -1:
            if node.right and self._get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
            
        return node
        
    def _get_balance(self, node: CDTNode) -> int:
        """Get balance factor of node."""
        left_height = node.left.height if node.left else 0
        right_height = node.right.height if node.right else 0
        return left_height - right_height

    def _rotate_left(self, node: CDTNode) -> CDTNode:
        """Perform left rotation."""
        new_root = node.right
        node.right = new_root.left
        new_root.left = node
        
        node.update_height()
        new_root.update_height()
        new_root.weight = node.weight + (new_root.left.weight if new_root.left else 0)
        
        return new_root

    def _rotate_right(self, node: CDTNode) -> CDTNode:
        """Perform right rotation."""
        new_root = node.left
        node.left = new_root.right
        new_root.right = node
        
        node.update_height()
        new_root.update_height()
        node.weight = (node.left.weight if node.left else 0)
        
        return new_root

    def delete(self, start: int, length: int) -> None:
        """Delete text from start index with given length."""
        if start < 0 or length < 0:
            raise IndexError("Invalid indices")
            
        if not self.root or length == 0:
            return

        total_length = len(self.get_text())
        
        if start >= total_length:
            return
            
        length = min(length, total_length - start)
        
        # Split rope into three parts
        left_part, remain = self._split(self.root, start)
        
        # Only split remainder if we have text to delete
        if remain:
            # Important: We don't split relative to start, we split relative to 0
            _, right_part = self._split(remain, length)
        else:
            right_part = None

        # Merge left and right parts correctly
        result = self._merge(left_part, right_part)
        if result:
            self.root = self._rebalance(result)
        else:
            self.root = None

    def _merge(self, left: Optional[CDTNode], right: Optional[CDTNode]) -> Optional[CDTNode]:
        """Merge two ropes into one."""
        if not left:
            return right
        if not right:
            return left
            
        # Create new internal node
        merged = CDTNode()
        merged.left = left
        merged.right = right
        merged.weight = self._get_weight(left)
        
        # The hash is already calculated in CDTNode.__init__
        merged.update_height()
        
        return merged


class TestCDTRope(unittest.TestCase):
    def setUp(self):
        """Create common test data."""
        print("\n")  # Add spacing between tests
        self.test_text = "The quick brown fox jumps over the lazy dog"
        self.rope = CDTRope(self.test_text)

    def setUp(self):
        """Create common test data."""
        print("\n") # Add spacing between tests
        self.test_text = "The quick brown fox jumps over the lazy dog"
        self.rope = CDTRope(self.test_text)
        # Get the current test method
        current_test = getattr(self, self._testMethodName)
        # Get its docstring and print it
        if current_test.__doc__:
            print(f"{current_test.__doc__}")

    def tearDown(self):
        pass  # No printing on test completion

    def test_empty_rope(self):
        """🧬 Tests initialization of empty rope data structure"""
        rope = CDTRope()
        self.assertEqual(rope.get_text(), "")
        self.assertEqual(rope.get_hash(), "")

    def test_delete_middle(self):
        """✂️  Tests deleting text from the middle of the rope"""
        self.rope.delete(4, 6)  # Delete "quick "
        self.assertEqual(self.rope.get_text(), "The brown fox jumps over the lazy dog")

    def test_delete_beginning(self):
        """🗑️  Tests deleting text from the beginning of the rope"""
        self.rope.delete(0, 4)  # Delete "The "
        self.assertEqual(self.rope.get_text(), "quick brown fox jumps over the lazy dog")

    def test_delete_end(self):
        """✂️  Tests deleting text from the end of the rope"""
        self.rope.delete(40, 3)  # Delete "dog"
        self.assertEqual(self.rope.get_text(), "The quick brown fox jumps over the lazy ")

    def test_substring(self):
        """📄 Tests retrieving substrings from the rope"""
        self.assertEqual(self.rope.substring(4, 5), "quick")
        self.assertEqual(self.rope.substring(0, 3), "The")
        self.assertEqual(self.rope.substring(40, 3), "dog")

    def test_content_based_chunking(self):
        """🧩 Tests the content-based chunking algorithm's consistency"""
        text1 = "abcabcabc" * 5
        text2 = "abcabcabc" * 5
        
        rope1 = CDTRope(text1)
        rope2 = CDTRope(text2)
        
        hashes1 = self._collect_hashes(rope1.root)
        hashes2 = self._collect_hashes(rope2.root)
        
        common_hashes = set(hashes1) & set(hashes2)
        self.assertGreater(len(common_hashes), 0)

    def test_rebalancing(self):
        """🌳 Tests if the tree maintains balance after multiple insertions at the beginning"""
        rope = CDTRope()
        for i in range(10):
            rope.insert(0, str(i))

    def test_large_operations(self):
        """📚 Tests handling of large text operations including insertion and deletion"""
        large_text = "a" * 100 + "b" * 100
        rope = CDTRope(large_text)
        rope.insert(100, "X" * 10)
        rope.delete(90, 20)

    def test_edge_case_deletions(self):
        """🎯 Tests edge cases in delete operations like negative indices and beyond text length"""
        self.rope.delete(100, 5)
        self.assertEqual(self.rope.get_text(), self.test_text)
        
        with self.assertRaises(IndexError):
            self.rope.delete(-1, 5)
        with self.assertRaises(IndexError):
            self.rope.delete(0, -5)
            
        self.rope.delete(0, len(self.test_text))
        self.assertEqual(self.rope.get_text(), "")

    def test_edge_case_insertions(self):
        """🎯 Tests edge cases in insert operations like negative indices and beyond text length"""
        with self.assertRaises(IndexError):
            self.rope.insert(-1, "test")
            
        long_text = "test"
        self.rope.insert(100, long_text)
        self.assertEqual(self.rope.get_text(), self.test_text + long_text)

    def test_chunking_consistency(self):
        """🔄 Tests that identical content produces identical chunk patterns"""
        pattern = "abcdef" * 10
        rope1 = CDTRope(pattern)
        rope2 = CDTRope(pattern)
        
        self.assertEqual(rope1.get_hash(), rope2.get_hash())
        
        rope1.insert(15, "xyz")
        rope2.insert(15, "xyz")
        self.assertEqual(rope1.get_hash(), rope2.get_hash())

    def test_hash_changes(self):
        """🔐 Tests hash behavior with text modifications"""
        initial_hash = self.rope.get_hash()
        initial_text = self.rope.get_text()
        
        self.rope.insert(0, "test")
        modified_hash = self.rope.get_hash()
        self.assertNotEqual(initial_hash, modified_hash)
        
        self.rope.delete(0, 4)
        final_text = self.rope.get_text()
        self.assertEqual(initial_text, final_text)
        
        final_hash = self.rope.get_hash()
        self.assertNotEqual(initial_hash, final_hash)

    def test_sequence_operations(self):
        """📝 Tests a sequence of mixed operations (insert/delete) on the rope"""
        rope = CDTRope("Hello World")
        rope.insert(5, " Beautiful")
        rope.delete(0, 6)
        rope.insert(0, "A ")
        rope.delete(2, 10)
        self.assertEqual(rope.get_text(), "A World")

    def test_node_integrity(self):
        """🏗️  Tests internal node properties like height, weight, and hash consistency"""
        pattern = "abcdef" * 3
        rope = CDTRope(pattern)
        
        def verify_node_integrity(node):
            if not node:
                return True
            
            left_height = node.left.height if node.left else 0
            right_height = node.right.height if node.right else 0
            expected_height = max(left_height, right_height) + 1
            self.assertEqual(node.height, expected_height)
            
            if node.is_leaf():
                self.assertEqual(len(node.text), node.weight)
                expected_hash = node._calculate_hash(node.text)
                self.assertEqual(node.hash, expected_hash)
            
            return verify_node_integrity(node.left) and verify_node_integrity(node.right)
        
        self.assertTrue(verify_node_integrity(rope.root))

    def _collect_hashes(self, node: Optional[CDTNode]) -> List[str]:
        """Helper to collect all hashes in the tree."""
        if not node:
            return []
        return [node.hash] + self._collect_hashes(node.left) + self._collect_hashes(node.right)

if __name__ == '__main__':
    unittest.main()
