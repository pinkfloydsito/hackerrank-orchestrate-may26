"""Ingestion pipeline: parse corpus, chunk, embed, and build FAISS index."""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from markdown_it import MarkdownIt
from tqdm import tqdm

from config import (
    DATA_DIR, FAISS_INDEX_PATH, METADATA_PATH, PRODUCT_AREAS_PATH,
    EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP, ECOSYSTEMS
)


def parse_yaml_frontmatter(text: str) -> tuple:
    """Extract YAML frontmatter from markdown text."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            # Simple YAML parsing
            yaml_text = parts[1].strip()
            metadata = {}
            for line in yaml_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"').strip("'")
            return metadata, parts[2].strip()
    return {}, text


def extract_headings(text: str) -> List[tuple]:
    """Extract headings and their positions from markdown."""
    headings = []
    for match in re.finditer(r'^(#{1,3}\s+.+)$', text, re.MULTILINE):
        level = len(match.group(1)) - len(match.group(1).lstrip('#'))
        title = match.group(1).lstrip('#').strip()
        headings.append((level, title, match.start()))
    return headings


def chunk_by_headings(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Split text into chunks by headings with overlap."""
    headings = extract_headings(text)
    
    if not headings:
        # No headings, split by paragraphs
        return chunk_by_paragraphs(text, chunk_size, overlap)
    
    chunks = []
    for i, (level, title, start_pos) in enumerate(headings):
        # Determine section end
        if i + 1 < len(headings):
            end_pos = headings[i + 1][2]
        else:
            end_pos = len(text)
        
        section_text = text[start_pos:end_pos].strip()
        
        # If section is too long, split it further
        words = section_text.split()
        if len(words) > chunk_size * 1.5:
            # Split into overlapping chunks
            for j in range(0, len(words), chunk_size - overlap):
                chunk_words = words[j:j + chunk_size]
                if len(chunk_words) < 20:  # Skip tiny chunks
                    continue
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "text": chunk_text,
                    "section_heading": title,
                })
        else:
            chunks.append({
                "text": section_text,
                "section_heading": title,
            })
    
    return chunks


def chunk_by_paragraphs(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Split text into chunks by paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for paragraph in paragraphs:
        words = paragraph.split()
        
        if current_word_count + len(words) > chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                "text": "\n\n".join(current_chunk),
                "section_heading": "",
            })
            
            # Start new chunk with overlap
            overlap_words = sum(len(p.split()) for p in current_chunk)
            while overlap_words > overlap and len(current_chunk) > 1:
                overlap_words -= len(current_chunk[0].split())
                current_chunk.pop(0)
            
            current_chunk = current_chunk[-1:] if current_chunk else []
            current_word_count = sum(len(p.split()) for p in current_chunk)
        
        current_chunk.append(paragraph)
        current_word_count += len(words)
    
    # Save final chunk
    if current_chunk:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "section_heading": "",
        })
    
    return chunks


def parse_corpus() -> tuple:
    """Parse all corpus files and return docs + product areas."""
    docs = []
    product_areas = set()
    
    for ecosystem in ECOSYSTEMS:
        ecosystem_dir = DATA_DIR / ecosystem
        if not ecosystem_dir.exists():
            print(f"Warning: {ecosystem_dir} not found")
            continue
        
        md_files = list(ecosystem_dir.rglob("*.md"))
        print(f"Found {len(md_files)} files in {ecosystem}")
        
        for md_file in tqdm(md_files, desc=f"Parsing {ecosystem}"):
            try:
                text = md_file.read_text(encoding="utf-8")
                metadata, body = parse_yaml_frontmatter(text)
                
                # Get article title from metadata or filename
                article_title = metadata.get("title", md_file.stem.replace("-", " ").replace("_", " "))
                
                # Extract headings for product area taxonomy
                headings = extract_headings(body)
                for level, title, _ in headings:
                    if level <= 2:  # H1 and H2
                        product_areas.add(title.lower())
                
                # Get relative path parts for product area inference
                rel_parts = md_file.relative_to(ecosystem_dir).parts
                if len(rel_parts) > 1:
                    # Use parent directory as product area hint
                    dir_hint = rel_parts[0].replace("-", " ").replace("_", " ")
                    product_areas.add(dir_hint.lower())
                
                # Chunk the document
                chunks = chunk_by_headings(body)
                
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "text": chunk["text"],
                        "product_ecosystem": ecosystem,
                        "source_file": str(md_file.relative_to(DATA_DIR)),
                        "article_title": article_title,
                        "section_heading": chunk.get("section_heading", ""),
                        "chunk_id": i,
                    })
                    
            except Exception as e:
                print(f"Error parsing {md_file}: {e}")
                continue
    
    return docs, sorted(list(product_areas))


def build_index(docs: List[Dict[str, Any]]) -> None:
    """Build FAISS index from documents."""
    print(f"\nEmbedding {len(docs)} chunks...")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["text"] for d in docs]
    
    # Encode in batches
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings).astype("float32")
    
    print(f"Building FAISS index with dimension {EMBEDDING_DIM}...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    
    print(f"Saving index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(docs, f)
    
    print(f"Index built successfully! Total chunks: {len(docs)}")


def main():
    """Main ingestion function."""
    print("=" * 60)
    print("CORPUS INGESTION PIPELINE")
    print("=" * 60)
    
    # Parse corpus
    docs, product_areas = parse_corpus()
    print(f"\nTotal chunks: {len(docs)}")
    print(f"Product areas found: {len(product_areas)}")
    
    # Save product areas
    with open(PRODUCT_AREAS_PATH, "w") as f:
        json.dump(product_areas, f, indent=2)
    print(f"Product areas saved to {PRODUCT_AREAS_PATH}")
    
    # Build and save index
    build_index(docs)
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
