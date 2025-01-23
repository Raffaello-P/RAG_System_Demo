import os
import yaml

from src.inference.rag_pipeline import RAGpipeline
from src.inference.vectorDB import Database


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")  # fornire path di configurazione
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("Configurazione in corso...")
    config = load_config()

    # Inizializzo la classe che contiene tutti i metodi utili alla pipeline implementata
    pipeline = RAGpipeline(config)

    # Inizializza database e crea tabella
    db = Database(config)

    # se bisogna inserire altri documenti
    if input("Vuoi eseguire l'inserimento di nuovi documenti nel database? (y/n): ").strip().lower() == 'y':
        #pipeline.chunks_division(db) # senza controllo sul numero dei token
        pipeline.chunk_division_llm(db) # con controllo sui token e riassunto con llm
    else:
        print("Operazione saltata.")


    print("Benvenuto! Digita una domanda per il modello (digita 'exit' per uscire).")
    while True:
        user_query = input(">> ").strip().lower()
        if user_query.lower() == "exit":
            print("Arrivederci!")
            break
        results_doc = pipeline.retriever.retrieve(user_query)
        print("Documenti di interesse recuperati")

        response = pipeline.query(user_query, results_doc)

        print(f"Risposta: {response['response']}\n\nRisposta basata sui seguenti documenti ritrovati:\n\n")

        for text, similarity in results_doc:
            print(f"{text} (similarità: {similarity:.2f})")


if __name__ == "__main__":
    main()
