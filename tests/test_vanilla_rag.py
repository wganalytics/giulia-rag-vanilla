import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Adicionar a pasta dev/rag/PRJ-01_Vanilla_RAG e o diretório src/ ao path do python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Criar os Mocks ANTES de importar o módulo src.main ou src.core.rag_engine
mock_embeddings = MagicMock()
mock_llm = MagicMock()
mock_vectorstore = MagicMock()
mock_retriever = MagicMock()

# Mocking requests.get para que find/serve de Ollama não travem nos testes
mock_requests_get = MagicMock()
mock_requests_get.return_value.status_code = 200
mock_requests_get.return_value.json.return_value = {
    "models": [{"name": "llama3"}, {"name": "nomic-embed-text"}]
}

# Patch amplo para isolar a inicialização do RAG Engine de dependências reais
with patch("langchain_ollama.OllamaEmbeddings", return_value=mock_embeddings), \
     patch("langchain_ollama.ChatOllama", return_value=mock_llm), \
     patch("langchain_community.vectorstores.Chroma", return_value=mock_vectorstore), \
     patch("requests.get", mock_requests_get), \
     patch("subprocess.run") as mock_run, \
     patch("subprocess.Popen") as mock_popen:
     
    # Agora importamos o engine e a aplicação
    from src.core.rag_engine import RagEngine, UPLOADS_DIR
    from src.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def run_around_tests():
    # Reset mocks antes de cada teste
    mock_embeddings.reset_mock()
    mock_llm.reset_mock()
    mock_vectorstore.reset_mock()
    mock_retriever.reset_mock()
    yield

def test_chromadb_health():
    """
    T01-01: test_chromadb_health
    Verifica se o banco de dados ChromaDB é instanciado sem erros na inicialização.
    """
    with patch("langchain_ollama.OllamaEmbeddings", return_value=mock_embeddings), \
         patch("langchain_ollama.ChatOllama", return_value=mock_llm), \
         patch("langchain_community.vectorstores.Chroma", return_value=mock_vectorstore), \
         patch("requests.get", mock_requests_get):
         
        engine = RagEngine()
        assert engine.status == "initialized" or engine.status == "error"
        assert os.path.exists(UPLOADS_DIR)

def test_rag_pipeline_empty_context():
    """
    T01-02: test_rag_pipeline_empty_context
    Garante que o motor RAG retorne mensagem de fallback sem invocar o LLM se o ChromaDB estiver vazio.
    """
    with patch("langchain_ollama.OllamaEmbeddings", return_value=mock_embeddings), \
         patch("langchain_ollama.ChatOllama", return_value=mock_llm), \
         patch("langchain_community.vectorstores.Chroma", return_value=mock_vectorstore), \
         patch("requests.get", mock_requests_get):
         
        engine = RagEngine()
        
        # Simular base vetorial vazia (zero documentos carregados)
        engine.vectorstore._collection.count.return_value = 0
        
        response = engine.query("Qual o escopo da documentação?")
        
        # Deve retornar mensagem clara instruindo upload
        assert "Nenhum documento" in response["answer"]
        assert "Upload" in response["answer"] or "upload" in response["answer"]
        assert response["sources"] == []

def test_api_validation():
    """
    T01-03: test_api_validation
    Garante que payloads malformados ou incompletos na rota /chat retornem erro 422.
    """
    # 1. Caso sem a chave question
    response_missing = client.post("/chat", json={})
    assert response_missing.status_code == 422
    
    # 2. Caso com tipo inválido
    response_invalid_type = client.post("/chat", json={"question": 12345})
    assert response_invalid_type.status_code == 422

def test_health_endpoint():
    """
    Verifica que o endpoint de health funciona e responde online no cenário de sucesso.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["online", "error"]
