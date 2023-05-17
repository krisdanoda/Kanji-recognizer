import bs4
import requests

def get_meaning(kanji):
    r = requests.get(f'https://jisho.org/search/{kanji}')
    r.raise_for_status()
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    meaningdiv = soup.find_all("div", {"class": "meanings english sense"})
    for element in meaningdiv:
        print(element.text)