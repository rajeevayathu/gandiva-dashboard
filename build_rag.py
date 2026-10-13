"""
build_rag.py — Index "Trade Like a Stock Market Wizard" into ChromaDB.

Run once (or after adding new books):
  /opt/homebrew/bin/python3.10 build_rag.py

Creates ./rag_db/ with the ChromaDB vector store.
Uses sentence-transformers (local, free, no API needed).
"""
import os, re, json
import pdfplumber
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DB   = os.path.join(BASE_DIR, 'rag_db')

BOOKS = [
    {
        'path':   '/Users/rajeevsharma/Downloads/Trade Like a Stock Market Wizard  (2013) - PDF Room.pdf',
        'title':  'Trade Like a Stock Market Wizard',
        'author': 'Mark Minervini',
        'skip_pages': 10,   # skip cover / TOC pages
    },
]

CHUNK_SIZE    = 600   # target chars per chunk (≈150 tokens)
CHUNK_OVERLAP = 100   # overlap between chunks to preserve context


def extract_pages(path: str, skip: int = 0) -> list[dict]:
    """Extract text from PDF, one dict per page."""
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            if i < skip:
                continue
            text = (page.extract_text() or '').strip()
            if len(text) < 50:   # skip near-blank pages
                continue
            pages.append({'page': i + 1, 'total': total, 'text': text})
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ''
    for sent in sentences:
        if len(current) + len(sent) <= size:
            current += (' ' if current else '') + sent
        else:
            if current:
                chunks.append(current.strip())
            # Start new chunk with overlap from end of previous
            words = current.split()
            overlap_text = ' '.join(words[-overlap//6:]) if words else ''
            current = (overlap_text + ' ' + sent).strip()
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 40]


def build():
    import chromadb
    from chromadb.config import Settings

    print('\n  AlphaEdge RAG Builder')
    print('  ─────────────────────────────────')

    # Load embedding model (downloads once, ~90MB, cached locally)
    print('  Loading embedding model (all-MiniLM-L6-v2)...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('  Model ready.')

    # Init ChromaDB
    client = chromadb.PersistentClient(path=RAG_DB)

    # Drop existing collection for clean rebuild
    try:
        client.delete_collection('minervini_books')
    except Exception:
        pass
    collection = client.create_collection(
        name='minervini_books',
        metadata={'hnsw:space': 'cosine'},
    )

    doc_id = 0
    for book in BOOKS:
        path = book['path']
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found — skipping.')
            continue

        print(f'\n  Processing: {book["title"]}')
        pages = extract_pages(path, skip=book.get('skip_pages', 0))
        print(f'  Extracted {len(pages)} pages with content.')

        all_chunks = []
        metadatas  = []
        ids        = []

        for page in pages:
            chunks = chunk_text(page['text'])
            for chunk in chunks:
                all_chunks.append(chunk)
                metadatas.append({
                    'book':   book['title'],
                    'author': book['author'],
                    'page':   page['page'],
                })
                ids.append(f'doc_{doc_id}')
                doc_id += 1

        print(f'  Created {len(all_chunks)} chunks. Embedding...')

        # Embed in batches of 64
        batch_size = 64
        for i in range(0, len(all_chunks), batch_size):
            batch_texts = all_chunks[i:i+batch_size]
            batch_meta  = metadatas[i:i+batch_size]
            batch_ids   = ids[i:i+batch_size]
            embeddings  = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            pct = min(100, int((i + batch_size) / len(all_chunks) * 100))
            print(f'  Embedded {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks ({pct}%)', end='\r')

        print(f'\n  Done: {len(all_chunks)} chunks indexed from {book["title"]}')

    total = collection.count()
    print(f'\n  RAG database ready at: {RAG_DB}')
    print(f'  Total chunks: {total}')

    # Save index metadata
    meta = {'books': [b['title'] for b in BOOKS], 'total_chunks': total}
    with open(os.path.join(BASE_DIR, 'rag_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('  Metadata saved to rag_meta.json\n')


if __name__ == '__main__':
    build()
