import bs4 as BeautifulSoup
import urllib.request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def create_dictionary_table(text_string) -> dict:
    # removing stop words 
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    # reducing words to their root form
    stem = PorterStemmer()
    # creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1
    return frequency_table

def main():
    # get a random wiki article
    fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/20th_century')
    article_read = fetched_data.read()

    # parsing the url 
    article_parsed = BeautifulSoup.BeautifulSoup(article_read, 'html.parser')

    # return p tags
    paragraphs = article_parsed.find_all('p')

    article_content = ''
    for p in paragraphs:
        article_content += p.text

    print(article_content)

if __name__ == "__main__":
    main()