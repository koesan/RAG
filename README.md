# RAG (Retrieval-Augmented Generation) 

Bu proje, Retrieval-Augmented Generation (RAG) yapısının nasıl kurulacağını ve uygulanacağını detaylandırmaktadır. RAG, bilgi getirme ve dil üretim süreçlerini birleştirerek daha anlamlı ve doğru yanıtlar elde edilmesini sağlar. Proje, Apache web loglarını içeren bir veri seti kullanarak RAG yapısının nasıl geliştirileceğini ve değerlendirileceğini kapsamlı bir şekilde ele alır. Verilerin ön işlenmesinden, embedding işlemlerine, vektör veri tabanı kullanımına ve sonuçların değerlendirilmesine kadar tüm adımlar detaylandırılmıştır.

# 0. Gerekli kurulumlar:

***main.py***
```
pip install langchain-community==0.2.12
pip install langchain==0.2.14
pip install user-agents==2.2.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
> [!NOTE]
> Benim bilgisayarımda CUDA sürümü 12.4 olduğu için, CUDA 12.4 ile uyumlu PyTorch sürümünü kurdum. Siz de kendi CUDA sürümünüz ile uyumlu olan PyTorch sürümünü indirmeniz gerekiyor. Bilgisayarınızda CUDA yoksa, CPU sürümünü indirebilirsiniz.
<br><br>

***local.py***
```
pip install transformers==4.44.0
pip install langchain==0.2.14
pip install user-agents==2.2.0
pip install langchain-huggingface==0.0.3
pip install langchain-community==0.2.12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
> [!NOTE]
> Benim bilgisayarımda CUDA sürümü 12.4 olduğu için, CUDA 12.4 ile uyumlu PyTorch sürümünü kurdum. Siz de kendi CUDA sürümünüz ile uyumlu olan PyTorch sürümünü indirmeniz gerekiyor. Bilgisayarınızda CUDA yoksa, CPU sürümünü indirebilirsiniz.

> [!NOTE]
> Kodu çalıştırdığınızda, öncelikle modellerin bilgisayarınıza indirilmesini beklemeniz gerekir. Bu süre, internet hızınıza bağlı olarak değişiklik gösterebilir. Eğer kod yazmak için Sublime Text gibi bir metin editörü kullanıyorsanız, bu indirme 
> işlemini göremezsiniz. Kodun uzun süre herhangi bir çıktı vermeden çalışması, indirme işlemlerinin devam ediyor olmasından kaynaklanabilir. Ancak, VS Code gibi kod yazma araçlarında terminal kısmında indirme durumu görülebilecektir.

> [!NOTE]
> Hem main.py hem de local.py dosyaları için bir Hugging Face API anahtarına ihtiyacınız var. Bunun için [Hugging Face](https://huggingface.co/) sitesine giriş yaptıktan sonra, sağ üst köşeden profilinize tıklayın, ardından Settings seçeneğine 
> gidin ve [Access Tokens](https://huggingface.co/settings/tokens) kısmına geçin. Gelen ekranda Create new token butonuna tıklayarak yeni bir token oluşturabilirsiniz. Bu token'ı kod içerisinde şu şekilde kullanmalısınız:
> 
> ``` os.environ["HUGGINGFACEHUB_API_TOKEN"] = " "``` 
<br><br>

---
## 1. Veri Seti Bulma:

Web trafik loglarını içeren veri setini bulmak için çeşitli kaynaklardan faydalandım. Bu kapsamda, Kaggle'da bulduğumuz ve Apache web sunucusu loglarını içeren veri setini tercih ettim. Bu veri seti, çeşitli log girdilerini kapsamlı bir şekilde içermektedir ve proje için uygun nitelikte verilere sahiptir.

**Veri Seti:** [Apache Web Log - Kaggle](https://www.kaggle.com/datasets/kimjmin/apache-web-log)

Bu veri seti, Apache web sunucusundan alınmış log kayıtlarını içerir ve IP adresleri, erişilen sayfalar, zaman damgaları gibi bilgileri içermektedir. Veri setinin içeriği, projede ihtiyaç duyduğumuz bilgileri sağlamaktadır.

> [!NOTE]
> Doğru veri seti seçimi, projeyi hazırlarken büyük önem taşır. Farklı bir veri seti üzerinde çalışma yaparken, veri setini yeterince incelemediğim için bu verilerin yapay zeka tarafından oluşturulmuş bir log dosyası olduğunu fark edemedim. 
> Veriler tutarsız ve tekrarlanıyordu, bu da ilk başta kurduğum modelin yanlış sonuçlar üretmesine neden oldu. Sorunun veri setinden kaynaklandığını anlamak zaman aldı ve bu nedenle çözüm süreci uzun sürdü.

---

## 2. Veri Ön İşleme:

Veri ön işleme, ham verilerin analiz ve modelleme için uygun hale getirilmesini sağlar. Bu süreçte, eksik, hatalı veya düzensiz veriler düzeltilir ve standartlaştırılır. Eksik veriler tamamlanır veya uygun şekilde işlenir, hatalı girişler düzeltilir ve farklı formatlardaki veriler tutarlı hale getirilir. Ayrıca, veriler normalizasyon ve standartlaştırma gibi tekniklerle uyumlu hale getirilir. Bu işlemler, verilerin doğruluğunu artırır ve modelin performansını iyileştirir, böylece elde edilen sonuçlar daha güvenilir ve anlamlı olur.

<br><br>
### 2.1 Verileri İnceleme:

Örnek bir log kaydı:
```
14.49.42.25 - - [12/May/2022:01:24:44 +0000] "GET /articles/ppp-over-ssh/ HTTP/1.1" 200 18586 "-" "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5
```

| Alan                          | Açıklama                                                                                   |
|-------------------------------|--------------------------------------------------------------------------------------------|
| **IP Adresi**                 | 14.49.42.25                                                                              |
| **Kullanıcı Kimliği**         | - - (Boş)                                                                                     |
| **Tarih ve Saat**             | [12/May/2022:01:24:44 +0000]                                                               |
| **HTTP İsteği**               | "GET /articles/ppp-over-ssh/ HTTP/1.1"                                                    |
| **Yanıt Durum Kodu**          | 200                                                                                       |
| **Veri Boyutu**               | 18586                                                                                     |
| **Yönlendiren URL**           | "-" (Boş)                                                                                 |
| **User-Agent**                | "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5; 16 subscribers; feed-id=3389821348893992437)" |


<br><br>

### 2.2 Kullanılacak Verileri Seçme:

Modelin daha iyi performans vere bilmesi için veri setindeki gereksiz verileri temizlemeliyiz. Bu nedenle, hangi verilerin kullanılabilir olduğuna karar verilmeli:

| Alan                              | Açıklama                                                                                                                               |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **IP Adresi (14.49.42.25)**       | IP adresi kullanılarak olası tehditler tespit edilebilir. Ancak bu projede gerekli olmadığını düşündüğümden dolayı IP adresini kullanmamaya karar verdim. |
| **- -**                           | Veri olmadığı için bu alanı sildim.                                                                                                     |
| **Tarih ve Saat ([12/May/2022:01:24:44 +0000])** | Veri setinde 12 Mayıs ile 7 Haziran arasındaki veriler bulunuyor. Model 17 Temmuz 2023'te güncellendiğinden, 2022 tarihli bilgilerin içeriyor olması cevabı etkileyebilir diye düşündüğümden almadım. |
| **HTTP İsteği ("GET /articles/ppp-over-ssh/ HTTP/1.1")** | Proje için anlamlı olmadığından kullanmamaya karar verdim.                                                   |
| **Durum Kodu ve Veri Boyutu (200, 18586)** | Sunucu yanıtı ve veri boyutunun proje için kullanmaya gerek olmadığını düşündüğümden bu verileri de temizledim.                     |
| **Otomatik Tarayıcılar ve Bozuk Veriler** | Bazı otomatik tarayıcılar, botlar veya bozuk veriler anlamlı bilgi içermediği için bunları da veri setinden çıkardım.                 |


Veriyi temizledikten sonra veriler aşağıdaki gibi olacak.

```
Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2b1) Gecko/20091014 Firefox/3.6b1 GTB5"; 16 subscribers; feed-id=3389821348893992437)
```
<br><br>
> [!NOTE]
> Bu veri üzerinde RAG yapısını kurup defalarca test ettiğimde sonuçların istediğim ölçüde başarılı olmadığını gözlemledim. Başarıyı artırmak için, öncelikle veri yapısını gözden geçirdim ve "user_agents" kütüphanesini kullanarak verileri 
> düzeltmeye karar verdim.
>
>"user_agents": User-Agent verilerini analiz etmek ve bu verilerilerden çeşitli bilgiler çıkarmak için kullanılan bir kütüphanedir.
<br><br>
user-agents kütüphanesi ile user-agents verilerindeki, istediğim verileri alarak veri setini düzelttim ve sonuç olarak aşağıdaki gibi bir veri yapsını elde ettim.

```
"Browser": "Firefox Beta", "Browser Version": "3.6.b1", "Operating System": "Windows"
```

Veri setini bu hale getirerek gereksiz verileri temizledim ve RAG yapısında gözle görlülür ölçüde İyileştirmeyi başardım.

> [!NOTE]
> Bu temizlik işlemlerinin tamamı useragents.py modülü içerisinde gerçekleştiriliyor

****

## 3. Sistemin Kurulumu:

RAG (Retrieval-Augmented Generation) iki ana yapıdan oluşur: Retrieval (Bilgi Getirme) ve Generation (Üretim).

### 3.1 Retrieval (Bilgi Getirme):

Retrieval katmanının temel amacı, kullanıcının sorduğu soruya en uygun verileri vektör veri tabanından getirmektir. Bu katman, sorgu ile veri tabanındaki veriler arasındaki benzerlikleri değerlendirir ve en ilgili bilgileri bulur. Sorgu, genellikle bir metin veya doğal dil ifadesi olarak sunulur ve bu ifade, vektörlere dönüştürülerek veri tabanındaki diğer vektörlerle karşılaştırılır. Bu karşılaştırma sonucunda, en yüksek benzerlik skoruna sahip veriler seçilir ve kullanıcıya sunulmak üzere üst katmanlara iletilir. Bu süreç, sistemin doğru ve anlamlı yanıtlar üretmesi için kritik bir adımdır ve bilgiye hızlı ve etkili erişim sağlar.

<br><br>
### 3.1.1 Verilerin İşlenmesi:

İlk adımda, log dosyasındaki verilerin bir vektör veri tabanına yüklenmesi gerekmektedir. Bu işlem için öncelikle veri setindeki veriler alınır, ardından uygun şekilde işlenir ve temizlenir. İşlenmiş veriler, daha sonra vektör formatına dönüştürülerek vektör veri tabanına eklenir. Bu adım, verilerin daha sonra yapılacak sorgulamalara hızlı ve etkili bir şekilde yanıt verebilmesi için gereklidir.

> [!NOTE]
> Useragents.py dosyası verileri weblog_sample dan çeker verileri temizler, uygun formata getirir ve bir liste olarak döndürür

LLM (Large Language Model) modelleri, belirli bir maksimum uzunlukta metin üzerinde çalışabilir. Bu uzunluğu aştığında, fazlalık kısmı görmezden gelir ve bu da sonuçların istenildiği gibi olmamasına ve modelin performans kaybına yol açar.
Bu nedenle, veri setindeki veriler LLM'nin işleyebileceği uzunluğa getirmek için her bir satırı listenin bir elemanı olarak ayarladım buda listedeki her indexte max 150, min 80 karakter olacağı anlamına geliyor.

`documents = useragents.process_log_files_to_list("weblog_sample.log")`

Bu listeyi Document nesnelerine dönüştürerek gereksiz boşlukları temizler ve boş satırları atlar. Bu sayede, işlenmiş veriler Document sınıfı formatında saklanır, bu da ilerideki işlemler için uygun bir yapı sağlar.

`documents = [Document(page_content=line.strip()) for line in documents if line.strip()]`

<br><br>
### 3.1.2 Embedding ile Vektörlere Dönüştürme:

Kesilen veriler, embedding işlemi ile sayısal vektörlere dönüştürülür. Bu süreçte, metinlerin anlamsal içeriği vektör uzayında temsil edilerek, benzerlik aramaları için uygun hale getirilir. Ardından, elde edilen vektörler FAISS Vektör Veri Tabanına yüklenir. Bu veri tabanı, yüksek boyutlu vektörler arasında hızlı ve etkili benzerlik aramaları yaparak, ilgili verilerin bulunmasını sağlar. Bu adım, özellikle büyük veri setlerinde verimli arama süreçleri için kritik öneme sahiptir.
```
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
    
embeddings = HuggingFaceEmbeddings()
    
book = FAISS.from_documents(documents, embeddings)
   
book.save_local("library") # İstenirse vektör veritabanı local bilgisayara kaydedilebilir.
```
> [!NOTE]
> Vektör veri tabanına yüklenen veriler lokalde kaydedilir. Bu sayede, sürekli olarak veri çekme, bölme ve vektörlere çevirme işlemleri tekrarlanmaz ve bu işlemlerle zaman kaybedilmez.

Embeddings modeli, RAG sisteminde kritik bir role sahiptir çünkü metinlerin anlamsal temsillerinin doğruluğunu belirler. Eğer embedding modeli güçlü ve etkiliyse, benzer anlamlara sahip metin parçaları vektör uzayında birbirine yakın vektörlerle temsil edilir. Bu, FAISS gibi vektör veri tabanlarında bu metin parçalarının doğru bir şekilde gruplandırılmasını sağlar. Böylece, sorguya en uygun veriler doğru konumlandırılır, bu da arama işleminin doğruluğunu ve etkinliğini garanti eder. Bu süreç, yalnızca sistemin performansını artırmakla kalmaz, aynı zamanda kullanıcıya daha isabetli sonuçlar sunar.

> [!NOTE]
> HuggingFaceEmbeddings modelini seçmemin nedenleri arasında güçlü API desteği, ücretsiz erişim ve yüksek performans yer alıyor.

<br><br>
### 3.3 Generation (Üretim):

RAG sisteminde kullanılan LLM (Büyük Dil Modeli) modelinin başarısı, üretilen sonuçların doğruluğunu ve anlamlılığını doğrudan etkiler. Uygun modelleri bulmak için "https://huggingface.co/" sayfasından arama yapabilirsiniz. 
Bu projede, performans ve uygunluk açısından en iyi seçenek olarak "google/flan-t5-large" modelini kullandım. 
Her ne kadar GPT gibi büyük dil modelleri oldukça başarılı olsa da, ücretli oldukları için bu projede kullanamadım. 
"google/flan-t5-xxl" gibi daha büyük modeller ise HuggingFaceHub APİ bağlantı sorunları nedeniyle tercih edilmedi(google/flan-t5-xxl Modelini local.py üzerinde çalıştırabilirsiniz). 
Sonuç olarak, performans ve erişilebilirlik açısından en uygun model olarak "google/flan-t5-large" modelinde karar kıldım.
```
from langchain_community.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.7, "max_length": 512})
```
Embedding kısmında yine ücretsiz ve kurulumu kolay olan HuggingFaceEmbeddings’i kullanmaya karar verdim. Bu sayede, projeyi kullanacak diğer insanlar için de pratik bir çözüm sunmuş oldum.
<br><br>

### 3.4 RAG Yapısının Çalıştırılması:

RAG yapısını çalıştırmak için öncelikle öncelikle veriler lokalde saklandıysa, localdeki verileri vektör veri setlerinin yüklenmesi gerekiyor:

`library = FAISS.load_local("book", embeddings, allow_dangerous_deserialization=True)`

Ardından, dil modeli ve bilgi getirme mekanizmasını bir araya getirmeliyiz. Böylece soru sorulduğunda, bilgi getirme mekanizması en uygun verileri arar ve dil modeli bu verilerle bir yanıt oluşturur.
```
chainSim = RetrievalQA.from_chain_type(
    llm=llm, # Büyük dil modeli
    chain_type="map_reduce", 
    retriever=library.as_retriever() # Vektör veri setinden  bilgi alma kısmı.   
)
```
<br><br>
> [!NOTE]
>chain_type: 
>    stuff: Basit ve hızlı bilgi getirme ve yanıt oluşturma için.
>    map_reduce: Büyük veri kümesi üzerinde paralel işleme için.
>    refine: İlk yanıtı iteratif olarak geliştirmek için.
>    map_rerank: Bilgi parçalarını sıralamak ve en uygun sonuçları seçmek için.
<br><br>

Son olarak, sorgumuzu yapıya aktararak çıktıyı alabiliriz:

`
chainSim.invoke(question)
`
> [!NOTE]
> Yapıyı kurarken çeşitli sorunlarla karşılaştım ama en çok zamanımı alan sorun, önceden çalışan kodun sonradan çalıştırmaya çalışırken sürekli hata vermesiydi. Ne kadar çalışırsamda bu hatayı çözemedim. Son olarak, vektör veritabanını 
> değiştirmeyi denediğimde sorunun çözüldüğünü fark ettim. Eğer kodunuz daha önce çalışıyorsa ama sonradan çalışmamaya başladıysa ve kodda bir sorun bulamıyorsanız, verilerinizi lokalde tutuyorsanız sorun büyük ihtimalle bu durumdadır. Kısacası, 
> vektör veritabanını silin, verileri tekrar işleyin ve yeniden kaydedin.
<br><br>

## 4. Değerlendirme:

Veri setini bulduktan, veriyi hazırladıktan ve RAG yapısını kurduktan sonra, en zorlu ve zaman alıcı aşama bu RAG yapısının doğruluğunu test etmek, yani ürettiği sonuçların ne kadar doğru ve ne kadar yanlış olduğunu belirlemektir.

Bu doğrulama işlemi için farklı yöntemler kullanılabilir, ancak temelde olan şey, modelin verdiği cevabın ne kadar doğru olduğunu değerlendirmektir. Bunun için iki yaklaşım önerilebilir:

1. Manuel Değerlendirme: Soruları elle sorarak, modelin verdiği cevapları doğruluğuna göre değerlendirirsiniz.

2. Otomatik Değerlendirme: Bir değerlendirme veri seti oluşturursunuz. Bu veri setinde her soru için doğru cevabı ve modelinizin verdiği cevabı tutarsınız. Daha sonra, farklı bir LLM modeliyle bu veri setini değerlendirip RAG yapısının doğruluğunu ölçebilirsiniz.

Bu değerlendirme sürecini daha objektif ve ölçülebilir kılmak için çeşitli metrikler kullanılır:

* Doğruluk (Accuracy): Modelin verdiği doğru cevapların toplam sorulara oranıdır.
* F1 Skoru: Modelin precision (kesinlik) ve recall (duyarlılık) değerlerinin harmonik ortalamasıdır. F1 Skoru, şu formülle hesaplanır: F1 Skoru = 2 * (precision * recall) / (precision + recall).
* Kullanıcı Geri Bildirimi: Kullanıcıların modelin verdiği cevaplarla ilgili memnuniyetini ölçerek de doğruluk değerlendirmesi yapılabilir.

Kendi projemde, modelin verdiği cevapları manuel olarak inceleyip doğruluğuna, verdiği cevaplara göre değerlendirerek bakmaya kararverdim. Bunun için 5 tane soru oluşturdum ve modele sordum ve cevaplarını aldım ayrıca aynı soruları mase modelede sordum ve onunda cevaplarını aldım.

| Sorgu                                                                                                             | Oluşturduğum RAG                           | Base LLM                   |
|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------|----------------------------|
| **What is the latest version of the Chrome browser used on MacOS X?**                                             | `35.0.1870` (Doğru)                       | `v 5.0` (Yanlış)           |
| **What is the version of the Firefox browser used on a Windows operating system with version 7?**                | `3.6` (Doğru)                             | `Firefox 7.0` (Yanlış)     |
| **What is the most widely used operating system on computers?**                                                   | `Windows` (Doğru)                         | `windows` (Doğru)          |
| **Which is the most used browser?**                                                                               | `IE` (Yanlış)                             | `google chrome` (Doğru)    |
| **How are missing data labeled?**                                                                                   | `Unknown` (Doğru)                         | `The following is a list of the ten most common sex-related disorders` (Yanlış) |

Sonuçları değerlendirdiğimizde, oluşturduğum RAG modeli 5 sorudan 4'üne istediğim cevapları vermeyi başardı; bu da benim için yeterli bir başarıydı. LLM modelinin log verileri ile ilgili sorularda, veri seti ile uyumsuz ve dolayısıyla yanlış cevaplar verdiği gözlemlendi. Ancak, genel olarak RAG yapısının başarılı sonuçlar ürettiği ve performansının tatmin edici olduğu sonucuna vardım.

## 5. Kalitesini artırma önerileri:

* Veri Seti: Daha kapsamlı bir veri seti seçilmesi ve veri temizliğinin daha detaylı yapılması, modelin başarısını artıracaktır.

* LLM: GPT gibi daha başarılı modellerin kullanılması, modelin çıktılarının kalitesini artıracak ve doğru sonuçlar elde edilmesini sağlayacaktır.

* Embeddings: Embeddings verilerinin vektör datasete yüklenmesi ve uygun verilerin modele verilmesi önemlidir. Daha iyi embeddings modellerinin kullanılması, sonuçları olumlu yönde etkileyecektir.

* Chain Type: stuff, map_reduce, refine, map_rerank gibi parametrelerle yapınıza uygun seçeneği kullanarak modelin daha iyi sonuçlar üretmesini sağlayabilirsiniz.

* as_retriever: search_type, top_k, filter gibi parametrelerle ince ayar yaparak yapının başarısını artırabilirsiniz.

* Prompt: İyi yazılmış bir prompt modelin doğru ve alakalı yanıtlar vermesini sağlar. Eksik veya belirsiz bir prompt, modelin yanlış veya alakasız yanıtlar üretmesine neden olur. Daha iyi promptlar yazarak daha başarılı sonuçlar alınabilinir.

> [!NOTE]
> HuggingFaceHub'da 10G model boyutu sınırlandırılması olması sebebi ile locak bilgisayarda isteidğiniz model ile çalıştıra bileceğiniz local.py kodunu projeye ekledim.
