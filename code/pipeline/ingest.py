"""Modern ingestion pipeline using LangChain + ChromaDB."""

import json
import os
import frontmatter
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    DATA_DIR, PRODUCT_AREAS_PATH, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, ECOSYSTEMS,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION
)


def parse_corpus() -> tuple[List[Any], set]:
    """Parse corpus using python-frontmatter and LangChain splitters."""
    all_chunks = []
    product_areas = set()
    
    # 1. Define how we split markdown
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    
    # Fallback token/character splitter for massive paragraphs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )

    for ecosystem in ECOSYSTEMS:
        ecosystem_dir = Path(DATA_DIR) / ecosystem
        if not ecosystem_dir.exists():
            print(f"Warning: {ecosystem_dir} not found")
            continue
        
        md_files = list(ecosystem_dir.rglob("*.md"))
        print(f"Found {len(md_files)} files in {ecosystem}")
        
        for md_file in tqdm(md_files, desc=f"Parsing {ecosystem}"):
            try:
                # 2. Safely parse frontmatter and body
                with open(md_file, "r", encoding="utf-8") as f:
                    post = frontmatter.load(f)
                
                metadata = post.metadata
                body = post.content
                
                article_id = metadata.get("article_id", md_file.stem)
                article_title = metadata.get("title", md_file.stem.replace("-", " "))
                
                # Extract product area from breadcrumbs (deterministic taxonomy)
                breadcrumbs = metadata.get("breadcrumbs", [])
                if breadcrumbs:
                    product_area = breadcrumbs[0]
                else:
                    # Fallback: use top-level directory or filename
                    rel_parts = md_file.relative_to(ecosystem_dir).parts
                    if len(rel_parts) > 1:
                        # File is in subdirectory - use directory name
                        product_area = rel_parts[0].replace("-", " ").replace("_", " ")
                    else:
                        # Root-level file - use filename stem (except index -> general)
                        stem = md_file.stem
                        product_area = "general" if stem.lower() == "index" else stem.replace("-", " ").replace("_", " ")
                
                product_areas.add(f"{ecosystem}:{product_area}")

                # 3. Split by Markdown Headers
                md_header_splits = markdown_splitter.split_text(body)
                
                # 4. Inject metadata into every section chunk
                for chunk in md_header_splits:
                    chunk.metadata.update({
                        "product_ecosystem": ecosystem,
                        "product_area": product_area,
                        "source_file": str(md_file.relative_to(DATA_DIR)),
                        "article_title": article_title,
                        "article_id": str(article_id),
                        "source_url": metadata.get("source_url", ""),
                        "breadcrumbs": " > ".join(breadcrumbs) if breadcrumbs else ""
                    })
                
                # 5. Fallback split to ensure chunks fit in embedding window
                final_splits = text_splitter.split_documents(md_header_splits)
                
                # 6. Clean metadata and add chunks
                for chunk in final_splits:
                    # Clean None values from metadata (ChromaDB rejects them)
                    chunk.metadata = {k: v for k, v in chunk.metadata.items() if v is not None}
                    all_chunks.append(chunk)

            except Exception as e:
                print(f"Error parsing {md_file}: {e}")
                continue
                
    return all_chunks, sorted(list(product_areas))


def build_index(chunks: List[Any]) -> None:
    """Upsert chunks into ChromaDB."""
    print(f"\nEmbedding and indexing {len(chunks)} chunks into ChromaDB...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Ensure persist directory exists
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    # Extract texts, metadatas, and generate deterministic IDs
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"doc-{i}" for i in range(len(chunks))]
    
    # Batch add/upsert to Chroma (larger batches for speed)
    batch_size = 500
    for i in tqdm(range(0, len(texts), batch_size), desc="Upserting to Chroma"):
        vectorstore.add_texts(
            texts=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        
    print(f"Index built successfully at {CHROMA_PERSIST_DIR}!")


def main():
    print("=" * 60)
    print("MODERN CORPUS INGESTION PIPELINE")
    print("=" * 60)
    
    chunks, product_areas = parse_corpus()
    print(f"\nTotal chunks generated: {len(chunks)}")
    print(f"Product areas found: {len(product_areas)}")
    
    with open(PRODUCT_AREAS_PATH, "w") as f:
        json.dump(product_areas, f, indent=2)
    print(f"Product areas saved to {PRODUCT_AREAS_PATH}")
    
    build_index(chunks)
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
