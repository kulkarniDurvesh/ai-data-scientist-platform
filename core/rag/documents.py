"""
Documents -> chunks that can be retrieved and cited.

Markdown is split at headings, so every chunk carries the path of
headings it sits under ("Doctor visit planning > Monthly plan"); long
sections are split at paragraph boundaries and tables stay whole.
Plain text files are split at blank lines.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 1200
MIN_CHARS = 40
SUFFIXES = (".md", ".txt")
# Files in a document folder that are settings, not documents.
RESERVED = {"examples.txt"}

STOPWORDS = set("""
a an the and or of to for in on at by with from into over under than then that this these those there
their them they it its is are was were be been being do does did have has had can could would should
will shall may might must i we you me my our your us what which who whom whose when where how why
whether if so as also just only per about between within each every all any some no not more most
less many much few than other such same own very too
""".split())


@dataclass
class Chunk:
    id: str
    source: str              # file path relative to its folder's parent
    title: str               # document title (first heading or file name)
    headings: list[str]      # heading path inside the document
    text: str
    position: int            # order inside the document

    @property
    def section(self) -> str:
        return " > ".join([self.title] + [h for h in self.headings if h != self.title])

    def searchable(self) -> str:
        """Text used for retrieval: the heading path gives context to short passages."""
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "title": self.title, "section": self.section, "text": self.text}


def words(text: str) -> list[str]:
    """Lowercase content words with a light stem ("visits" -> "visit")."""

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def _stem(word: str) -> str:
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    lines = text.replace("\r\n", "\n").split("\n")
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), Path(source).stem.replace("_", " "))

    sections: list[tuple[list[str], list[str]]] = []
    path: list[str] = []
    body: list[str] = []
    for line in lines:
        heading = re.match(r"^(#{1,4})\s+(.*)", line)
        if heading:
            if "".join(body).strip():
                sections.append((list(path), body))
            level = len(heading.group(1))
            path = path[: level - 1] + [heading.group(2).strip()]
            body = []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((list(path), body))

    chunks: list[Chunk] = []
    for headings, body_lines in sections:
        for piece in _split(body_lines):
            if len(piece) < MIN_CHARS and chunks and chunks[-1].headings == headings:
                chunks[-1].text += "\n\n" + piece
                continue
            chunks.append(Chunk("", source, title, headings, piece, len(chunks)))

    for chunk in chunks:
        chunk.id = hashlib.sha1(f"{source}|{chunk.position}|{chunk.text}".encode()).hexdigest()[:12]
    return chunks


def _split(lines: list[str]) -> list[str]:
    """Paragraphs (tables and lists kept together), packed up to MAX_CHARS."""

    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    pieces: list[str] = []
    for block in blocks:
        if pieces and len(pieces[-1]) + len(block) + 2 <= MAX_CHARS:
            pieces[-1] += "\n\n" + block
        else:
            pieces.append(block)
    return [p for p in pieces if p.strip()]


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    fingerprint: str = ""


def load_corpus(paths: list[Path]) -> Corpus:
    """All .md / .txt files under the given folders (or single files)."""

    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(
                p for p in path.rglob("*")
                if p.suffix.lower() in SUFFIXES and p.is_file() and p.name.lower() not in RESERVED
            ))

    corpus = Corpus()
    digest = hashlib.sha1()
    for file in files:
        stat = file.stat()
        digest.update(f"{file}|{stat.st_size}|{stat.st_mtime_ns}".encode())
        source = _display_path(file)
        text = file.read_text(encoding="utf-8", errors="replace")
        corpus.chunks.extend(chunk_markdown(text, source))
        corpus.files.append(source)
    corpus.fingerprint = digest.hexdigest()
    return corpus


def _display_path(file: Path) -> str:
    parts = file.resolve().parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else file.name
