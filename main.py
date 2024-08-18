import os
import useragents
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFace API token'ını ortam değişkeni olarak ayarla
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Dil modeli (LLM) ayarla
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large", # Kullanılacak model
    model_kwargs={"temperature": 0.7, "max_length": 512} # Modelin parametreleri
)

# Embedding modeli ayarla
embeddings = HuggingFaceEmbeddings()

# Log dosyasından belgeleri al
documents = useragents.process_log_files_to_list("weblog_sample.log")

# Boş satırları filtrele ve verileri modele uygun olan formata çevir.
documents = [Document(page_content=line.strip()) for line in documents if line.strip()]

# FAISS kullanarak belge vektörlerini oluştur
library = FAISS.from_documents(documents, embeddings)

# FAISS kütüphanesini yerel olarak kaydetme seçeneği (isteğe bağlı olarak)
# library.save_local("bert_base_uncased_2")

# Soru belirle
question = "What is the most commonly used browser on Android?"

# FAISS kütüphanesini yerel olarak yükleme seçeneği (isteğe bağlı olarak)
# library = FAISS.load_local("bert_base_uncased", embeddings, allow_dangerous_deserialization=True)

# RetrievalQA zincirini oluştur
chainSim = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce", # Kullanılacak zincir tipi
    retriever=library.as_retriever(search_type="mmr")
)
print("---------------------")
print(chainSim.invoke(question))
print("---------------------")
