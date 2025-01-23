from src.preprocessing.chunkDivisionOllama import chunkDivisionOllama
from src.preprocessing.chunkSplit import pdf_to_chunks_with_langchain
from src.inference.ollama_client import OllamaClient
from src.preprocessing.docEmbedding import EmbeddingGenerator
from src.inference.retrieval import Retriever
from src.inference.vectorDB import Database
import os

class RAGpipeline:
    def __init__(self, config, max_token = 500, overlap = 50):
      # Inserire nome del file in self.document_path di seguito
        self.config = config
        self.llm_client = OllamaClient(config["llm"])
        self.document_path = os.path.join(
            os.path.dirname(__file__),
            config["paths"]["documents_dir"],
            "FileName.docx"
        )
        self.documents_dir = os.path.join(os.path.dirname(__file__), config["paths"]["documents_dir"])
        self.prompt_template = config["prompt"]
        self.prompt_preprocessing = config["prompt_preprocessing"]
        self.max_token = max_token
        self.overlap = overlap
        self.embedder = EmbeddingGenerator(self.config["embedding_model"])  # inizializzo l'embedder
        self.retriever = Retriever(config) # Inizializzo il Retriever per la ricerca semantica dei documenti simili alla query


    def chunks_division(self, db: Database):
        #s e devo inserire altri documenti a db (compreso di embedding)
        # divido i documenti in chunks
        chunks_text, chunks_metadata = pdf_to_chunks_with_langchain(pdf_path=self.document_path)  # facoltativi i campi max_tokens=200, chunk_overlap=50)

        # Genera embedding e popola il database, lancio in altra sessione
        embeddings = self.embedder.generate_embeddings(chunks_text)  # genero gli embeddings del testo (di tutti i chunks)

        # inserisco tutti i dati a db
        db.insert_vectors(chunks_text, chunks_metadata, embeddings)


    def chunk_division_llm(self, db: Database):
        # La seguente porzione di codice è da sostituire alle prossime solo nel caso del mio progetto personale (non considerare)
        #chunkOllama = chunkDivisionOllama_colloquio(self.document_path, self.config["embedding_model"]["model_name"],
        #                                 max_tokens=self.max_token, chunk_overlap=self.overlap)
        # Split del documento
        # chunks, metadata = chunkOllama.chuncks_division_personalizzato()

        chunkOllama = chunkDivisionOllama(self.document_path, self.config["embedding_model"]["model_name"],
                                          max_tokens=self.max_token, chunk_overlap=self.overlap)
        # Split del documento
        chunks, metadata = chunkOllama.chuncks_division()
        # Preprocessing dei chunk: se troppo lunghi vengono riassunti da un llm
        final_chunks, modified_index_chunks = chunkOllama.chunks_dim_elaboration(chunks, self.config, self.llm_client)
        print(f"Sono stati riassunti i chunk numero: {modified_index_chunks}")

        # Genera embedding e popola il database, lancio in altra sessione
        embeddings = self.embedder.generate_embeddings(final_chunks)  # genero gli embeddings del testo (di tutti i chunks)
        # inserisco tutti i dati a db
        db.insert_vectors_db2(chunksinfo=metadata, chunkstext=final_chunks, embeddings=embeddings)


    def query(self, user_query, simDoc = ""):
        context = simDoc
        #context_text = "\n".join([doc['content'] for doc in context])
        full_prompt = self.prompt_template.format(context=context, query=user_query)

        return self.llm_client.query(full_prompt)
