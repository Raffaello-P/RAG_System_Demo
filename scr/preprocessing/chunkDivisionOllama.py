import re
import docx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import os
from src.inference.ollama_client import OllamaClient


class chunkDivisionOllama:
    def __init__(self,docx_path, tokenizer_model, max_tokens=512, chunk_overlap=50):
        self.docx_path = docx_path
        self.file_extension = os.path.splitext(docx_path)[-1].lower()
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.metadata = os.path.basename(docx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)


    def set_docx_path(self, new_path):
        self.docx_path = new_path
        self.metadata = os.path.basename(self.docx_path)


    def chuncks_division(self):
        doc = docx.Document(self.docx_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text.strip())
        content = "\n".join(text)

        # Trova tutte le sezioni basate sul pattern '1)', '2)', etc.
        pattern = r"(^\d+\))|(\n\d+\))"  #
        matches = list(re.finditer(pattern, content))

        # Se non ci sono match, restituisci tutto il testo come un unico chunk
        if not matches:
            return [content], [{"source_chunk_index": 0}]

        # Suddividi in sezioni basate sui match
        chunks = []
        metadata = []
        for i in range(len(matches)):
            start = matches[i].end()  # Fine del match corrente
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)  # Inizio del prossimo match o fine del documento
            chunk = content[start:end].strip()
            chunks.append(f"{matches[i].group(0)} {chunk}")
            metadata.append(self.metadata)

        return chunks, metadata


    def chunks_dim_elaboration(self, chunks, config, clientLLM: OllamaClient):
        final_chunks = []
        prompt_template = config['prompt_preprocessing']
        modified_index_chunks = []
        # se il numero di token è maggiore di quello consentito allora fai un riassunto
        for i, chunk in enumerate(chunks):
            if len(self.tokenizer.encode(chunk)) > self.max_tokens:
                full_prompt = prompt_template.format(context=chunk)
                response = clientLLM.preprocessingQuery(full_prompt)
                final_chunks.append(response["response"])
                modified_index_chunks.append(i)
            else:
                final_chunks.append(chunk)

        return final_chunks, modified_index_chunks

    def pdf_to_chunks_with_langchain(self):
        # Carica il documento PDF usando LangChain
        loader = PyPDFLoader(self.docx_path)
        documents = loader.load()

        # Estrai il testo e utilizza un TextSplitter per il chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens * 4,  # Circa 4 caratteri per token (approssimazione)
            chunk_overlap=50,  # Sovrapposizione tra chunk per preservare il contesto
            separators=[". ", "? ", "! "]  # Spezza solo alla fine di una frase
        )

        # Suddividi il testo in chunk
        chunks = text_splitter.split_documents(documents)

        # Converte i chunk in una lista di stringhe
        chunk_texts = [chunk.page_content for chunk in chunks]  # testo senza metadati

        return chunk_texts
