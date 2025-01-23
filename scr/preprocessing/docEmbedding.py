from sentence_transformers import SentenceTransformer
import yaml

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(self.config['model_name'])

    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
