from user_agents import parse
import re

def parse_user_agent(user_agent):
    # Useragent string'ini parse etmek için user_agents kütüphanesini kullan
    ua = parse(user_agent)
    
    # Useragent içindeki dil bilgisini düzenli ifade ile çek
    lang_match = re.search(r'\b([a-z]{2}-[A-Z]{2})\b', user_agent)
    language = lang_match.group(1) if lang_match else 'Unknown'  # Dil bulunamazsa 'Unknown' olarak ayarla

    # Useragent verilerini bir sözlük olarak döndür
    return {
        'Browser': ua.browser.family or 'Unknown',  # Tarayıcı adı
        'Browser Version': ua.browser.version_string or 'Unknown',  # Tarayıcı versiyonu
        'Operating System': ua.os.family or 'Unknown',  # İşletim sistemi
        'Operating System Version': ua.os.version_string or 'Unknown',  # İşletim sistemi versiyonu
        'Language': language  # Dil bilgisi
    }

def process_log_files_to_list(input_log_file):
    seen_entries = set()  # Görülen girişleri takip etmek için küme
    results = []  # Sonuçları depolamak için liste
    
    with open(input_log_file, 'r') as infile:
        for line in infile:
            # Her satırda Userqgent bilgilerini çekmek için düzenli ifade kullan
            user_agent_match = re.search(r'\"(Mozilla[^"]+)\"$', line.strip())
            if user_agent_match:
                user_agent = user_agent_match.group(1)  
                result = parse_user_agent(user_agent)  
                
                # Sonuçları formatlayarak bir string oluştur
                formatted_result = (
                    f"Browser: {result['Browser']}, "
                    f"Browser Version: {result['Browser Version']}, "
                    f"Operating System: {result['Operating System']}, "
                    f"Operating System Version: {result['Operating System Version']}, "
                    f"Language: {result['Language']}"
                )

                results.append(formatted_result)  # Formatlanmış sonucu listeye ekle
    return results
