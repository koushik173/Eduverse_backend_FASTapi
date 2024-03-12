import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests

API_KEY = 'AIzaSyDs_4G9ieEvR7y5qQMKWZTKrst-Ua6676c'
SEARCH_ENGINE_ID = 'd7f3353e4543149d3'

def search_text(query):
    url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}'
    response = requests.get(url)
    if response.status_code == 200:
        search_results = response.json()
        urls = [item['link'] for item in search_results.get('items', [])]
        return urls
    else:
        print("Error key:", response.status_code)
        return []

def get_web_page_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching content from {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

def compare_texts(original_text, web_texts):
    vectorizer = CountVectorizer().fit_transform([original_text] + web_texts)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    similarities = similarity_matrix[0][1:]
    return similarities

def find_sources(original_text, urls, thress):
    web_texts = []
    for url in urls:
        web_content = get_web_page_content(url)
        if web_content:
            web_text = extract_text_from_html(web_content)
            web_texts.append(web_text)

    similarities = compare_texts(original_text, web_texts)
    sources = {url: similarity for url, similarity in zip(urls, similarities) if similarity >= thress}

    return sources