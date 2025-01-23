import psycopg2
from psycopg2.extras import execute_values
import json


class Database:
    def __init__(self, config):
        self.config = config
        self.conn = psycopg2.connect(
            host = self.config['vector_db']['DB_HOST'],
            port = self.config['vector_db']['DB_PORT'],
            dbname = self.config['vector_db']['DB_NAME'],
            user = self.config['vector_db']['DB_USER'],
            password = self.config['vector_db']['DB_PASSWORD']
        )
        self.vector_table = config['vector_db']['DB_VECTOR_TABLE']

    def create_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS %s
        (
            id integer NOT NULL DEFAULT nextval('documents_id_seq'::regclass),
            content text COLLATE pg_catalog."default",
            embedding vector(768),
            chunkinfo json,
            CONSTRAINT documents_pkey PRIMARY KEY (id)
        )
        
        TABLESPACE pg_default;
        
        ALTER TABLE IF EXISTS %s
            OWNER to postgres;
        
        COMMENT ON COLUMN public.%s."chunkMetadata"
            IS 'metadati del chunk in formato json';
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (self.vector_table,self.vector_table))
            self.conn.commit()

    def insert_vectors(self, chunks_texts, chunks_metadata, embeddings):
        """
        Inserisce i vettori nel database, mantenendo la corrispondenza 1 a 1
        tra chunks_texts, chunks_metadata ed embeddings.

        Parametri:
            chunks_texts (list): Lista di testi dei chunk.
            chunks_metadata (list): Lista di metadati associati ai chunk.
            embeddings (list): Lista di embedding associati ai chunk.
        """
        query = f"INSERT INTO {self.vector_table} (content, embedding, chunkinfo) VALUES %s"

        # Prepara i dati per l'inserimento
        data = [
            (chunks_texts[i], embeddings[i].tolist(), json.dumps(chunks_metadata[i]))
            for i in range(len(chunks_texts))
        ]

        # Esegui l'inserimento
        try:
            with self.conn.cursor() as cur:
                execute_values(cur, query, data)
                self.conn.commit()
            print("Insert avvenuta con successo")
        except Exception as e:
            # Se si verifica un errore, effettua il rollback della transazione
            self.conn.rollback()
            print(f"Errore nella insert: {e}")
            raise e  # Rilancia l'errore per ulteriori analisi

    def insert_vectors_db2(self, chunksinfo, chunkstext, embeddings):
        """
        Inserisce i vettori nel database, mantenendo la corrispondenza 1 a 1
        tra chunks_texts, chunks_metadata ed embeddings.

        Parametri:
            chunkinfo: Lista di metadati associati ai chunk.
            chunktext:
            embeddings: Lista di embedding associati ai chunk.
        """
        query = f"INSERT INTO {self.vector_table} (chunkinfo, chunktext, embedding) VALUES %s"

        # Prepara i dati per l'inserimento
        data = [
            (json.dumps(chunksinfo[i]), chunkstext[i], embeddings[i].tolist())
            for i in range(len(chunkstext))
        ]

        # Esegui l'inserimento
        try:
            with self.conn.cursor() as cur:
                execute_values(cur, query, data)
                self.conn.commit()
            print("Insert avvenuta con successo")
        except Exception as e:
            # Se si verifica un errore, effettua il rollback della transazione
            self.conn.rollback()
            print(f"Errore nella insert: {e}")
            raise e  # Rilancia l'errore per ulteriori analisi

    def search_vectors(self, query_embedding, top_k):
        query = f"""
        SELECT chunktext, 1 - (embedding <=> %s::vector) AS similarity
        FROM {self.vector_table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        try:
            with self.conn.cursor() as cur:
                # Passiamo query_embedding come un array di float, ma lo castiamo come tipo vector
                cur.execute(query, (query_embedding, query_embedding, top_k))
                return cur.fetchall()
        except Exception as e:
            # Se si verifica un errore, effettua il rollback della transazione
            self.conn.rollback()
            print(f"Errore nella query: {e}")
            raise e  # Rilancia l'errore per ulteriori analisi
