from src.inference.vectorDB import Database
from src.preprocessing.docEmbedding import EmbeddingGenerator

class Retriever:
    def __init__(self, config):
        self.config = config
        self.db = Database(config)
        self.embedder = EmbeddingGenerator(config["embedding_model"])

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        query_embedding = self.embedder.generate_embeddings([query])[0]
        return self.db.search_vectors(query_embedding.tolist(), top_k)
