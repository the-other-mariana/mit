import bs4 as BeautifulSoup
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

class ExtractionSummary:

    def __init__(self, text):
        self.source_text = text

    def summarize(self):
        ft = self.create_dictionary_table(self.source_text)
        sentences = sent_tokenize(self.source_text)
        sw = self.calculate_sentence_scores(sentences, ft)
        avg_score = self.calculate_avg_score(sw)
        summary = self.extract(sentences, sw, avg_score)
        return summary

    def create_dictionary_table(self) -> dict:
        # removing stop words 
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(self.source_text)
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

    def calculate_sentence_scores(self, sentences, frequency_table) -> dict:
        # algorithm for scoring a sentence by its words
        # to avoid the whole sentence being the dict key, the key is just the 1st 7 chars of each sentence
        sentence_weight = dict()
        for sentence in sentences:
            sentence_wordcount = (len(word_tokenize(sentence)))
            sentence_wordcount_without_stop_words = 0
            for word_weight in frequency_table:
                if word_weight in sentence.lower():
                    sentence_wordcount_without_stop_words += 1
                    if sentence[:7] in sentence_weight:
                        sentence_weight[sentence[:7]] += frequency_table[word_weight]
                    else:
                        sentence_weight[sentence[:7]] = frequency_table[word_weight]
            sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words
        return sentence_weight

    def calculate_avg_score(self, sentence_weights) -> int:
        # avg score for the sentences, to avoid choosing sentences with score < avg score for the summary
        sum_values = 0
        for weight in sentence_weights:
            sum_values += sentence_weights[weight]
        avg_score = (sum_values / len(sentence_weights))
        return avg_score

    def extract(self, sentences, sentence_weights, threshold):
        sentence_counter = 0
        summary = ''
        for sentence in sentences:
            key = sentence[:7]
            if key in sentence_weights and sentence_weights[key] >= threshold:
                summary += " " + sentence
                sentence_counter += 1
        return summary


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

    nltk.download('stopwords')
    nltk.download('punkt')
    
    extraction_summary = ExtractionSummary(article_content)
    summary = extraction_summary.summarize()

    print("Summary:")
    print(summary)


if __name__ == "__main__":
    main()