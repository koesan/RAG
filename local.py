import os
import useragents
from transformers import pipeline
from langchain.schema import Document
from langchain_chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Hugging Face Hub API token'ını ortam değişkeni olarak ayarla
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Modeli tanımla
model_id = "google/flan-t5-xl"
# Tokenizer ve modeli yükle
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
# Model ve tokenizer ile bir pipeline oluştur
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)

# pipe'ı HuggingFacePipeline dönüştür.
llm = HuggingFacePipeline(pipeline=pipe)

# Embeddingsi tanımla
embeddings = HuggingFaceEmbeddings()

# Boş satırları filtrele ve verileri modele uygun olan formata çevir.
documents = useragents.process_log_files_to_list("weblog_sample.log")
documents = [Document(page_content=line.strip()) for line in documents if line.strip()]

# FAISS kullanarak belge vektörlerini oluştur
library = FAISS.from_documents(documents, embeddings)

# FAISS kullanarak belge vektörlerini oluştur
library = FAISS.from_documents(documents, embeddings)

# FAISS kütüphanesini yerel olarak kaydetme seçeneği (isteğe bağlı olarak)
# library.save_local("bert_base_uncased")

# Soru belirle
question = "What is the most commonly used browser on Android?"

# FAISS kütüphanesini yerel olarak yükleme seçeneği (isteğe bağlı olarak)
# library = FAISS.load_local("bert_base_uncased", embeddings, allow_dangerous_deserialization=True)

# RetrievalQA zincirini oluştur
chainSim = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  
    retriever=library.as_retriever(search_type="mmr") 
)

print("---------------------")
print(chainSim.invoke(question))
print("---------------------")
