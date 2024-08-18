# RAG

## 1. Veri Seti Bulma:

Web trafik loglarını içeren veri setini bulmak için çeşitli kaynaklardan faydalandım. Bu kapsamda, Kaggle'da bulduğumuz ve Apache web sunucusu loglarını içeren veri setini tercih ettim. Bu veri seti, çeşitli log girdilerini kapsamlı bir şekilde içermektedir ve proje için uygun nitelikte verilere sahiptir.

- **Kullanılan Veri Seti**: [Apache Web Server Access Logs](https://www.kaggle.com/datasets/kimjmin/apache-web-log)

Bu veri seti, Apache web sunucusundan alınmış log kayıtlarını içerir ve IP adresleri, erişilen sayfalar, zaman damgaları gibi bilgileri içermektedir. Veri setinin içeriği, projede ihtiyaç duyduğumuz bilgileri sağlamaktadır.

---

## 2. Veri Ön İşleme:

Veri setini seçtikten sonra, öncelikle verilerin içeriğini incelememiz gerekir.

### 2.1 Verileri İnceleme:

Örnek bir log kaydı:

\`\`\`
14.49.42.25 - - [12/May/2022:01:24:44 +0000] "GET /articles/ppp-over-ssh/ HTTP/1.1" 200 18586 "-" "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5"
\`\`\`
	
Bu kayıtta yer alan bilgiler:

- IP Adresi**: `14.49.42.25`
- Tarih ve Saat**: `[12/May/2022:01:24:44 +0000]`
- HTTP İsteği**: `"GET /articles/ppp-over-ssh/ HTTP/1.1"`
- Durum Kodu ve Veri Boyutu**: `200 18586`
- User-Agent Bilgileri**: `"Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5"`

Farklı log dosyalarında farklı veriler olabilir.

### 2.2 Kullanılacak Verileri Seçme:

Modelin daha iyi performans vere bilmesi için veri setindeki gereksiz verileri temizlemeliyiz. Bu nedenle, hangi verilerin kullanılabilir olduğuna karar verilmeli:

- **IP Adresi (14.49.42.25): IP adresi kullanılarak olası tehditler tespit edilebilir. Ancak bu projede gerekli olmadığını düşündüğümden dolayı IP adresini kullanmamaya karar verdim.
- **-*-: Veri olmadığı için bu alanı sildim.
- **Tarih ve Saat ([12/May/2022:01:24:44 +0000]): 12 Mayıs ile 7 Haziran arasındaki veriler bulunuyor. Model 17 Temmuz 2023'te güncellendiğinden, 2022 tarihli bilgilerin içeriyor olması cevabı etkiliyeceğini düşündüğümdne almadım.
- **HTTP İsteği ("GET /articles/ppp-over-ssh/ HTTP/1.1"): Proje için anlamlı olmadığından kullanmamaya karar verdim.
- **Durum Kodu ve Veri Boyutu (200, 18586): Sunucu yanıtı ve veri boyutunun proje için kullanmaya gerek olmadığını düşündüğümden bu verileri de temizledim.
- **Otomatik Tarayıcılar ve Bozuk Veriler: Bazı otomatik tarayıcılar, botlar veya bozuk veriler anlamlı bilgi içermediği için bunları da veri setinden çıkardım.

Bu işlemler yapıldıktan sonra aşağıdaki gibi bir veri yapısı oldu.
\`\`\`
Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5"; 16 subscribers; feed-id=3389821348893992437)
\`\`\`

Bu veri üzerinde RAG yapısını kurup defalarca test ettiğimde sonuçların istediğim ölçüde başarılı olmadığını gözlemledim. Başarıyı artırmak için öncelikle veri yapsını elden geçridim ve "user_agents" kütüphanesini kullanarak verileri düzeltmeye kararvedim.

"user_agents": User-Agent dizgilerini analiz etmek ve bu dizgilerden çeşitli bilgiler çıkarmak için kullanılan bir kütüphanedir.

bu kütüphane ile user-agents verilerindeki istediğim verileri alarak veri setini düzelttim ve sonuç olarak aşağıdaki gibi bir veri yapsını elde ettim.

	"Browser": "Firefox Beta", "Browser Version": "3.6.b1", "Operating System": "Windows"

Bu verilerde kullanılan tarayıcı, tarayıcı versiyonu, işletim sistemi, sistem dili bilgileri etiketli bir şekilde kullanıla bilecek.

Veri setini bu hale getirmek RAG yapısında gözle görlülür ölçüde İyileştirmeyi başardım.

***Not: Bu temizlik işlemlerinin tamamı useragents modülü içerisinde geçekleştirilecek***

Sonuç olarak, elimde şu şekilde temizlenmiş bir veri kaldı:

	"Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5"; 16 subscribers; feed-id=3389821348893992437)"

Elimdeki veri setinde 300.000 satır veri vardı temizleme işleminden sonra 229561 satır veri kadlı.

****

## 3. Sistemin Kurulumu

RAG (Retrieval-Augmented Generation) iki ana yapıdan oluşur: Retrieval (Bilgi Getirme) ve Generation (Üretim).

### Retrieval (Bilgi Getirme):

Bu katmanın esas amacı, vektör veri tabanında soruya en uygun verileri getirmektir.

### 3.1 Verilerin İşlenmesi:

İlk adımda, veri setindeki veriler alınır ve işlenir.

***Useragents.py dosyası verileri weblog_sample dan çeker verileri temizler, uygun formata getirir ve bir liste olarak döndürür.***

LLM (Large Language Model) modelleri, belirli bir maksimum uzunlukta metin üzerinde çalışabilir. Bu uzunluğu aştığında, fazlalık kısmı görmezden gelir ve bu da sonuçların istenildiği gibi olmamasına ve modelin performans kaybına yol açar.
Bu nedenle, veri setindeki veriler LLM'nin işleyebileceği uzunluğa getirmek için her bir satırı listenin bir elemanı olarak ayarladım buda listedeki her indexte max 150, min 80 karakter olacağı anlamına geliyor.

	documents = useragents.process_log_files_to_list("weblog_sample.log")

Bu listeyi Document nesnelerine dönüştürerek gereksiz boşlukları temizler ve boş satırları atlar. Bu sayede, işlenmiş veriler Document sınıfı formatında saklanır, bu da ilerideki işlemler için uygun bir yapı sağlar.

	documents = [Document(page_content=line.strip()) for line in documents if line.strip()]

### 3.2 Embedding ile Vektörlere Dönüştürme:

Kesilen veriler, embedding işlemi ile sayısal vektörlere dönüştürülür. Bu sayede metinlerin anlamsal içeriği vektör uzayında temsil edilir ve benzerlik aramaları bu vektörler üzerinde yapılır. Ardından, elde edilen vektörler FAISS Vektör Veri Tabanına yüklenir.

	from langchain_community.vectorstores import FAISS
	from langchain_community.embeddings import HuggingFaceEmbeddings
	
	embeddings = HuggingFaceEmbeddings()
	
	book = FAISS.from_documents(documents, embeddings)
	
	Vektör veri tabanına yüklenen veriler lokalde kaydedilir. Bu sayede sürekli veri çekme, bölme ve vektörlere çevirme işlemleri tekrarlanmaz ve sorgulama işlemleri hızlanır.
	
	book.save_local("library") # İstenirse vektör veritabanı local bilgisayara kaydedilebilir.

Embeddings modeli, RAG sisteminde kritik bir role sahiptir. Metinlerin anlamsal temsillerinin ne kadar doğru yapıldığını belirler. Eğer embedding modeli güçlü ve etkiliyse, benzer anlamlara sahip metin parçaları vektör uzayında birbirine yakın vektörlerle temsil edilir. 
Bu da, FAISS gibi vektör veri tabanlarında bu metin parçalarının doğru bir şekilde gruplandırılmasını sağlar. Bu süreç, yalnızca performansı artırmakla kalmaz, aynı zamanda sorguya en uygun verilerin doğru konumlandırılmasıyla arama işleminin doğruluğunu ve etkinliğini de garanti eder. 

HuggingFaceEmbeddings modelini seçmemin nedenleri arasında güçlü API desteği, ücretsiz erişim ve yüksek performans yer alıyor.

### 3.3 Generation (Üretim):

RAG sisteminde kullanılan LLM modelinin başarısı, üretilecek sonuçların doğruluğunu ve anlamlılığını doğrudan etkiler. Uygun modeller için "https://huggingface.co/" sayfasından arama yapabilirsiniz. Bu projede, en uygun model olan "google/flan-t5-large" modelini kullandım. GPT gibi büyük dil modelleri her ne kadar başarılı olsa da, ücretli oldukları için kullanamadım. 
"google/flan-t5-xxl" gibi modeller ise API bağlantı sorunları nedeniyle tercih etmedim. Performans açısından en uygun model olarak "google/flan-t5-large" modelinde karar kıldım.

	from langchain_community.llms import HuggingFaceHub
	
	os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
	
	llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.7, "max_length": 512})

Embedding kısmında yine ücretsiz ve kurulumu kolay olan HuggingFaceEmbeddings’i kullanmaya karar verdim. Bu sayede, projeyi kullanacak diğer insanlar için de pratik bir çözüm sunmuş oldum.

RAG Yapısının Çalıştırılması

RAG yapısını çalıştırmak için öncelikle öncelikle veriler lokalde saklandıysa, localdeki verileri vektör veri setlerinin yüklenmesi gerekiyor:

	library = FAISS.load_local("book", embeddings, allow_dangerous_deserialization=True)

Ardından, dil modeli ve bilgi getirme mekanizmasını bir araya getirmeliyiz. Böylece soru sorulduğunda, bilgi getirme mekanizması en uygun verileri arar ve dil modeli bu verilerle bir yanıt oluşturur.

	chainSim = RetrievalQA.from_chain_type(
	    llm=llm,
	    chain_type="map_reduce", 
	    retriever=library.as_retriever()		
	)

	chain_type: 
		stuff: Basit ve hızlı bilgi getirme ve yanıt oluşturma için.
		map_reduce: Büyük veri kümesi üzerinde paralel işleme için.
		refine: İlk yanıtı iteratif olarak geliştirmek için.
		map_rerank: Bilgi parçalarını sıralamak ve en uygun sonuçları seçmek için.


Son olarak, sorgumuzu yapıya aktararak çıktıyı alabiliriz:

	chainSim.invoke(question)

## 4. Değerlendirme.

Veri setini bulduktan, veriyi hazırladıktan ve RAG yapısını kurduktan sonra, en zorlu ve zaman alıcı aşama bu RAG yapısının doğruluğunu test etmek, yani ürettiği sonuçların ne kadar doğru ve ne kadar yanlış olduğunu belirlemektir.

Bu doğrulama işlemi için farklı yöntemler kullanılabilir, ancak temelde önemli olan şey, modelin verdiği cevabın ne kadar doğru olduğunu değerlendirmektir. Bu doğrulama için iki yaklaşım önerilebilir:

1. Manuel Değerlendirme: Soruları elle sorarak, modelin verdiği cevapları doğruluğuna göre değerlendirirsiniz.

2. Otomatik Değerlendirme: Bir değerlendirme veri seti oluşturursunuz. Bu veri setinde her soru için doğru cevabı ve modelinizin verdiği cevabı tutarsınız. Daha sonra, farklı bir LLM modeliyle bu veri setini değerlendirip RAG yapısının doğruluğunu ölçebilirsiniz.

Bu değerlendirme sürecini daha objektif ve ölçülebilir kılmak için çeşitli metrikler kullanılır:

	Doğruluk (Accuracy): Modelin verdiği doğru cevapların toplam sorulara oranıdır.
	F1 Skoru: Modelin precision (kesinlik) ve recall (duyarlılık) değerlerinin harmonik ortalamasıdır. F1 Skoru, şu formülle hesaplanır: F1 Skoru = 2 * (precision * recall) / (precision + recall).
	Kullanıcı Geri Bildirimi: Kullanıcıların modelin verdiği cevaplarla ilgili memnuniyetini ölçerek de doğruluk değerlendirmesi yapılabilir.

Kendi projemde, modelin verdiği cevapları manuel olarak inceleyip, doğruluğunu değerlendirerek karar verdim. Bunun için 5 tane soru oluşturdum ve modele sordum ve cevaplarını aldım ayrıca aynı soruları mase modelede sordum ve onunda cevaplarını aldım.

					Oluşturduğum RAG 														       Base LLM																									
	{'query': 'What is the latest version of the Chrome browser used on MacOS X?', 'result': '35.0.1870', Doğru}				  				 v 5.0  (Yanlış)
	{'query': 'What is the version of the Firefox browser used on a Windows operating system with version 7?', 'result': '3.6', Doğru}			   	      Firefox 7.0  (Yanlış)
	{'query': 'What is the most widely used operating system on computers?', 'result': 'Windows', Doğru}								       	 windows (Doğru)
	{'query': 'Which is the most used browser?', 'result': 'IE', Yanlış}												      google chrome (Doğru)
	{'query': 'How are missing data labeled?', 'result': 'Unknown', Doğru}									 The following is a list of the ten most common sex-related disorders (Yanlış)

Sonuçları değerlendirdiğimizde, LLM modeli, log verilerinin kullanılması gerektiği sorularda tahmin edilebildiği gibi veri seti ile uyuşmayan, kısaca yanlış cevaplar verdiği gözlemlenmiştir. Sonuç olarak, RAG yapısı gözle görülür şekilde başarılı sonuçlar üretmektedir.

## 5. Başarıyı artırma önerileri:

* Veri Seti: Daha kapsamlı bir veri seti seçilmesi ve veri temizliğinin daha detaylı yapılması, modelin başarısını artıracaktır.

* LLM: GPT-4 gibi daha başarılı modellerin kullanılması, modelin çıktılarının kalitesini artıracak ve doğru sonuçlar elde edilmesini sağlayacaktır.

* Embeddings: Embeddings verilerinin vektör datasete yüklenmesi ve uygun verilerin modele verilmesi önemlidir. Daha iyi embeddings modellerinin kullanılması, sonuçları olumlu yönde etkileyecektir.

* Chain Type: stuff, map_reduce, refine, map_rerank gibi parametrelerle yapınıza uygun seçeneği kullanarak modelin daha iyi sonuçlar üretmesini sağlayabilirsiniz.

* as_retriever: search_type, top_k, filter gibi parametrelerle ince ayar yaparak yapının başarısını artırabilirsiniz.


***Nont: local.py Rag modelini istediğiniz LLM modeli ile kendi bilgisayarınızca çalıştırabilmeniz için. (HuggingFaceHub'da 10G model boyut sınırlaması mevcut)***
