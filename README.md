# RAG_System_Demo
Demo di un sistema RAG. In questo repo sono disponibili le principali classi e metodi implementati per la crazione e gestione dell'intera pipeline di un sistema RAG.

# Introduzione
Questo progetto contiene una demo di un sistema RAG da me implementato. Qui è possibile trovare il necessario per poter personalizzare una propria versione. 
Per inserire nuovi documenti, metterli nella cartella documents ed inserire il nome del documento in rag_pipeline.py come da commento. 
Dopo aver fatto la suddivisione in chunks e l'embedding (e dopo aver salvato tutto a db), si consiglia di spostare i documenti o il documento nella cartella old_documents.

# Componenti fondamentali
1. Ollama
2. PostgreSQL -> pgvector
3. Creare un db contenente una tabella in grado di gestire vettori (serve da punto 1 pgvector per la ricerca semantica)
4. Personalizzare il file config con le giuste informazioni (modelli da utilizzare, percorsi, ecc...)
