import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Simple tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, documents):
        """
        documents: list of strings
        """
        self.documents = documents

        print(f"Building BM25 index for {len(documents):,} documents...")

        tokenized_documents = [
            tokenize(doc)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print("BM25 index built.")

    def retrieve(self, query, k=5):
        """
        Retrieve top-k documents for a query.
        """
        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[-k:][::-1]

        results = [
            self.documents[i]
            for i in top_indices
        ]

        return results