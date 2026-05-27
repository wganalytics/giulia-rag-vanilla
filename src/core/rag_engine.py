import os
import subprocess
import time
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import requests
from dotenv import load_dotenv

# Carrega configurações do .env se existir
load_dotenv()


def ensure_ollama_running():
    """Verifica se Ollama está rodando, se não, inicia."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=2)
        if response.status_code == 200:
            print("[RAG] Ollama já está rodando.")
            return True
    except Exception:
        pass
    
    print("[RAG] Iniciando Ollama...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        for _ in range(10):
            try:
                response = requests.get(f"{ollama_host}/api/tags", timeout=2)
                if response.status_code == 200:
                    print("[RAG] Ollama iniciado com sucesso.")
                    return True
            except Exception:
                time.sleep(1)
        
        print("[RAG] Erro ao iniciar Ollama.")
        return False
    except Exception as e:
        print(f"[RAG] Erro ao iniciar Ollama: {e}")
        return False


def ensure_model_loaded(model_name: str):
    """Verifica se o modelo especificado está disponível, se não, fazpull."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", m.get("model", "")) for m in models]
            
            if any(model_name in m for m in model_names):
                print(f"[RAG] Modelo '{model_name}' já disponível.")
                return True
            
            print(f"[RAG] Baixando modelo '{model_name}'...")
            result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[RAG] Modelo '{model_name}' baixado com sucesso.")
                return True
            else:
                print(f"[RAG] Erro ao baixar modelo: {result.stderr}")
                return False
    except Exception as e:
        print(f"[RAG] Erro ao verificar modelo: {e}")
        return False

# Configurações do RAG vindas das variáveis de ambiente ou com valores padrão de fallback
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text") 
LLM_MODEL = os.getenv("MODEL_NAME", "llama3") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Descobrindo a raiz do projeto (três níveis acima de src/core/rag_engine.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

class RagEngine:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        if not ensure_ollama_running():
            print("[RAG] AVISO: Ollama não está disponível. A aplicação pode não funcionar corretamente.")
        
        if not ensure_model_loaded(LLM_MODEL):
            print(f"[RAG] AVISO: Modelo '{LLM_MODEL}' não pôde ser carregado.")
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            self.llm = ChatOllama(
                model=LLM_MODEL, 
                temperature=0.1,
                base_url=OLLAMA_BASE_URL
            )
            self.vectorstore = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=self.embeddings
            )
            # Retriver com MMR para melhor diversidade
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 5, "fetch_k": 10}
            )
            
            # Setup da Chain (Corrente Langchain)
            template = """
            Você é um assistente prestativo que responde perguntas baseado APENAS no contexto fornecido.
            Leia attentamente o contexto abaixo e responda a pergunta de forma clara e direta.
            Se a resposta não estiver no contexto, diga: "Não encontrei essa informação no documento."
            Não invente respostas.

            Contexto do documento:
            {context}

            Pergunta: {question}

            Resposta:
            """
            prompt = PromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
                
            self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            self.status = "initialized"
        except Exception as e:
            print(f"Erro ao inicializar RAG Engine: {e}")
            self.status = "error"

    def process_pdf(self, file_path: str):
        """Lê um PDF, divide em chunks e salva no VectorDB."""
        if not os.path.exists(file_path):
            raise FileNotFoundError("PDF file not found.")

        # Carregar com PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Split em Chunks Menores
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # Adicionar ao Banco de Dados (Chroma)
        self.vectorstore.add_documents(documents=splits)
        return len(splits)

    def query(self, question: str):
        """Faz uma pergunta à chain de RAG."""
        if self.vectorstore._collection.count() == 0:
            return {"answer": "Nenhum documento foi processado ainda. Faça upload de um PDF primeiro.", "sources": []}
            
        print(f"Buscando contexto para a pergunta: {question}")
        response = self.rag_chain.invoke(question)
        
        source_docs = self.retriever.invoke(question)
        sources = [f"Página {doc.metadata.get('page', 'N/A')} do arquivo {os.path.basename(doc.metadata.get('source', 'Desconhecido'))}" for doc in source_docs]
        
        return {
            "answer": response,
            "sources": list(set(sources))
        }

    def clear_database(self):
        """Remove todos os documentos do VectorDB e deleta arquivos de upload."""
        try:
            # Deletar coleção no Chroma
            self.vectorstore.delete_collection()
            # Reinicializar o vectorstore
            self.vectorstore = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=self.embeddings
            )
            # Reconfigurar o retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 5, "fetch_k": 10}
            )
            
            # Reconfigurar a chain
            template = """
            Você é um assistente prestativo que responde perguntas baseado APENAS no contexto fornecido.
            Leia attentamente o contexto abaixo e responda a pergunta de forma clara e direta.
            Se a resposta não estiver no contexto, diga: "Não encontrei essa informação no documento."
            Não invente respostas.

            Contexto do documento:
            {context}

            Pergunta: {question}

            Resposta:
            """
            prompt = PromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
                
            self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Limpar pasta de uploads
            for file in os.listdir(UPLOADS_DIR):
                file_path = os.path.join(UPLOADS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            print("[RAG] Banco de dados e uploads limpos com sucesso.")
            return True
        except Exception as e:
            print(f"[RAG] Erro ao limpar banco de dados: {e}")
            return False

    def list_documents(self):
        """Retorna uma lista de nomes de arquivos extraídos diretamente do VectorDB."""
        try:
            # 1. Buscar metadados de todos os fragmentos no banco
            all_data = self.vectorstore._collection.get()
            
            # 2. Extrair o nome do arquivo (basename) de cada 'source'
            sources = set()
            for metadata in all_data['metadatas']:
                source_path = metadata.get('source', '')
                if source_path:
                    sources.add(os.path.basename(source_path))
            
            return sorted(list(sources))
        except Exception as e:
            print(f"[RAG] Erro ao listar documentos do DB: {e}")
            return []

    def remove_document(self, filename: str):
        """Remove um documento específico do VectorDB e apaga o arquivo físico."""
        try:
            full_path = os.path.join(UPLOADS_DIR, filename)
            
            # 1. Buscar todos os IDs onde o 'source' termina com o filename
            # Isso resolve problemas onde o caminho absoluto mudou no disco (ex: renomear pasta do projeto)
            all_data = self.vectorstore._collection.get()
            ids_to_delete = [
                all_data['ids'][i] 
                for i, metadata in enumerate(all_data['metadatas']) 
                if metadata.get('source', '').endswith(filename)
            ]
            
            if ids_to_delete:
                self.vectorstore._collection.delete(ids=ids_to_delete)
                print(f"[RAG] {len(ids_to_delete)} fragmentos do documento '{filename}' removidos do VectorDB.")
            else:
                print(f"[RAG] Nenhum fragmento encontrado para '{filename}' no VectorDB.")
            
            # 2. Remover Arquivo Físico
            if os.path.exists(full_path):
                os.remove(full_path)
            
            return True
        except Exception as e:
            print(f"[RAG] Erro ao remover documento '{filename}': {e}")
            return False

# Instância Singleton Engine
rag_engine_instance = RagEngine()
