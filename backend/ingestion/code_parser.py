"""
Parse source files into semantically meaningful chunks (functions/classes)
instead of arbitrary fixed-size windows. Falls back to paragraph-based
chunking for plain text / unsupported languages.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from tree_sitter_languages import get_parser
    TREE_SITTER_AVAILABLE = True
except Exception:
    TREE_SITTER_AVAILABLE = False

EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".cs": "c_sharp", ".php": "php", ".rs": "rust",
}

# Node types that represent a "chunkable" unit across common grammars.
CHUNK_NODE_TYPES = {
    "function_definition", "function_declaration", "method_definition",
    "class_definition", "class_declaration", "function_item", "impl_item",
}

MAX_CHUNK_CHARS = 3000
MIN_CHUNK_CHARS = 40


@dataclass
class CodeChunk:
    file_path: str
    function_name: Optional[str]
    line_start: int
    line_end: int
    text: str
    language: str


def _extract_name(node, source: bytes) -> Optional[str]:
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
    return None


def _chunk_with_tree_sitter(file_path: str, source: bytes, lang: str) -> List[CodeChunk]:
    parser = get_parser(lang)
    tree = parser.parse(source)
    chunks: List[CodeChunk] = []

    def walk(node):
        if node.type in CHUNK_NODE_TYPES:
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            if len(text.strip()) >= MIN_CHUNK_CHARS:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    function_name=_extract_name(node, source),
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    text=text[:MAX_CHUNK_CHARS],
                    language=lang,
                ))
            return  # don't descend into nested functions to avoid duplicate overlap
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def _chunk_plain_text(file_path: str, text: str, language: str = "text") -> List[CodeChunk]:
    """Fallback: split by blank-line paragraphs, grouped up to MAX_CHUNK_CHARS."""
    lines = text.splitlines()
    chunks: List[CodeChunk] = []
    buf, start_line = [], 1
    char_count = 0

    for i, line in enumerate(lines, start=1):
        buf.append(line)
        char_count += len(line)
        if char_count >= MAX_CHUNK_CHARS or i == len(lines):
            chunk_text = "\n".join(buf).strip()
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    function_name=None,
                    line_start=start_line,
                    line_end=i,
                    text=chunk_text,
                    language=language,
                ))
            buf, char_count, start_line = [], 0, i + 1
    return chunks


def parse_file(file_path: str) -> List[CodeChunk]:
    """Parse a single file into chunks, choosing the best available strategy."""
    ext = Path(file_path).suffix.lower()
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    lang = EXT_TO_LANG.get(ext)

    if lang and TREE_SITTER_AVAILABLE:
        try:
            chunks = _chunk_with_tree_sitter(file_path, raw, lang)
            if chunks:
                return chunks
        except Exception:
            pass  # fall through to plain text

    text = raw.decode("utf-8", errors="ignore")
    return _chunk_plain_text(file_path, text, language=lang or "text")


def parse_files(file_paths: List[str]) -> List[CodeChunk]:
    all_chunks: List[CodeChunk] = []
    for fp in file_paths:
        all_chunks.extend(parse_file(fp))
    return all_chunks
