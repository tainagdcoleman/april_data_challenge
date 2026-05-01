"""Print the contents of the local ChromaDB."""

import argparse
import json

import chromadb


def main():
    parser = argparse.ArgumentParser(description="Inspect contents of a local ChromaDB.")
    parser.add_argument("--path", default="./chroma_db", help="Path to the persistent Chroma directory.")
    parser.add_argument("--collection", default=None, help="Only show this collection (default: all).")
    parser.add_argument("--limit", type=int, default=5, help="Max items to print per collection (0 = all).")
    parser.add_argument("--full", action="store_true", help="Print full document text (no truncation).")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.path)
    collections = client.list_collections()

    if args.collection:
        collections = [c for c in collections if c.name == args.collection]
        if not collections:
            print(f"No collection named {args.collection!r} found.")
            return

    if not collections:
        print(f"No collections found at {args.path}.")
        return

    for col_info in collections:
        col = client.get_collection(col_info.name)
        total = col.count()
        print(f"\n=== Collection: {col.name}  (count={total}) ===")

        if total == 0:
            continue

        get_kwargs = {"include": ["documents", "metadatas"]}
        if args.limit > 0:
            get_kwargs["limit"] = args.limit
        data = col.get(**get_kwargs)
        for i, (_id, doc, meta) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
            print(f"\n[{i}] id={_id}")
            print(f"    metadata: {json.dumps(meta, ensure_ascii=False)}")
            if doc is not None:
                text = doc if args.full else (doc[:300] + ("..." if len(doc) > 300 else ""))
                print(f"    document: {text}")

        if args.limit > 0 and total > args.limit:
            print(f"\n  (showing {args.limit} of {total} — pass --limit 0 for all)")


if __name__ == "__main__":
    main()
