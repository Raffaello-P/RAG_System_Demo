import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# nltk.download('punkt_tab') # Scarica il pacchetto per il tokenizing (vedi README)
nltk.download('punkt')  # Scarica il pacchetto per il tokenizing (vedi README)


def pdf_to_chunks(pdf_path, max_tokens=200):
    # Estrai il testo dal PDF
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    # Tokenizza in frasi
    sentences = sent_tokenize(full_text)

    # Dividi in chunk rispettando i limiti di token
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        tokens_in_sentence = word_tokenize(sentence)
        token_count = len(tokens_in_sentence)

        # Se non ho ancora raggiunto il numero di token massimo consentito
        if current_token_count + token_count <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += token_count
        else:
            # Se ho raggiunto il numero max completa il chunk attuale e inizia uno nuovo
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_token_count = token_count

    # Aggiungi l'ultimo chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def pdf_to_chunks_with_langchain(pdf_path, max_tokens=200, chunk_overlap=50):
    # Carica il documento PDF usando LangChain
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Estrai il testo e utilizza un TextSplitter per il chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # Circa 4 caratteri per token (approssimazione)
        chunk_overlap=chunk_overlap,  # Sovrapposizione tra chunk per preservare il contesto
        separators=[". ", "? ", "! "]  # Spezza solo alla fine di una frase
    )

    # Suddividi il testo in chunk
    chunks = text_splitter.split_documents(documents)

    # Converte i chunk in una lista di stringhe
    chunks_texts = [chunk.page_content for chunk in chunks]  # testo senza metadati
    chunks_metadata = [chunk.metadata for chunk in chunks]

    return chunks_texts, chunks_metadata
