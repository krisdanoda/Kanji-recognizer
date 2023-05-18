import bs4
import requests

def get_meaning(kanji):
    r = requests.get(f'https://jisho.org/search/{kanji}')
    r.raise_for_status()
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    meaning_div = soup.find_all("div", {"class": "meanings english sense"})
    meaning_list = []
    for element in meaning_div:
        meaning_list.append(element.text)
    
    return meaning_list