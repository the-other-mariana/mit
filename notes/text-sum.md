# Text Summarizing

Types of text summarization:

- Extraction-based summarization

In extraction-based summarization, a subset of words that represent the most important points is pulled from a piece of text and combined to make a summary.

> In machine learning, extractive summarization usually involves weighing the essential sections of sentences and using the results to generate summaries.

- Abstraction-based summarization

Advanced deep learning techniques are applied to paraphrase and shorten the original document, just like humans do.

> Since abstractive machine learning algorithms can generate new phrases and sentences that represent the most important information from the source text, they can assist in overcoming the grammatical inaccuracies of the extraction techniques.

How to perform text summarization?

Here's a sample text:

> “Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital. Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well. Therefore, Peter stayed with her at the hospital for 3 days without leaving.”

1. Convert the paragraph into sentences

- The best way of doing the conversion is to extract a sentence whenever a period appears.

2. Text processing

Let’s do text processing by removing the stop words (extremely common words with little meaning such as “and” and “the”), numbers, punctuation, and other special characters from the sentences.

Basic text processing:

- Text cleaning for de-noising the text with nltk library's set of stopwords (words that add no value to the text as they appear in most texts, ie, "that", "the", etc).

- The PorterStemmer algorithm, which reduces words into their root form, ie, "cleaning" and "cleaned" becomes "clean".

3. Tokenization

Tokenizing the sentences is done to get all the words present in the sentences. Here is a list of the words:

```
['peter','elizabeth','took','taxi','attend','night','party','city','party','elizabeth','collapse','rush','hospital', 'diagnose','brain', 'injury', 'doctor','told','peter','stay','besides','get','well','peter', 'stayed','hospital','days','without','leaving']
```

4. Evaluate the weighted occurrence frequency of the words

To achieve this, let’s divide the occurrence frequency of each of the words by the frequency of the most recurrent word in the paragraph, which is “Peter” that occurs three times.

| WORD |	FREQUENCY |	WEIGHTED FREQUENCY |
| --- | --- | --- |
| peter	| 3	| 1 |
| elizabeth	| 2	| 0.67 |
| took	| 1	| 0.33 |
| taxi	| 1	| 0.33 |
| attend	| 1	| 0.33 |
| night	| 1	| 0.33 |
| party	| 2	| 0.67 |
| city	| 1	| 0.33 |
| collapse	| 1	| 0.33 |
| rush	| 1	| 0.33 |
| hospital	| 2	| 0.67 |
| diagnose	| 1	| 0.33 |
| brain	| 1	| 0.33 |
| injury	| 1	| 0.33 |
| doctor	| 1	| 0.33 |
| told	| 1	| 0.33 |
| stay	| 2	| 0.67 |
| besides	| 1	| 0.33 |
| get	| 1 | 	0.33 |
| well	| 1	| 0.33 |
| days	| 1	| 0.33 |
| without	| 1	| 0.33 |
| leaving	| 1	| 0.33 |

5. Substitute words with their weighted frequencies

Let’s substitute each of the words found in the original sentences with their weighted frequencies. Then, we’ll compute their sum.

Since the weighted frequencies of the insignificant words, such as stop words and special characters, which were removed during the processing stage, is zero, it’s not necessary to add them.

| SENTENCE	| ADD WEIGHTED FREQUENCIES	| SUM	| RESULT |
| --- | --- | --- | --- |
| 1	 | Peter and Elizabeth took a taxi to attend the night party in the city	| 1 + 0.67 + 0.33 + 0.33 + 0.33 + 0.33 + 0.67 + 0.33	| 3.99
| 2	| While in the party, Elizabeth collapsed and was rushed to the hospital	| 0.67 + 0.67 + 0.33 + 0.33 + 0.67	| 2.67
| 3	| Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well.	| 0.33 + 0.33 + 0.33 + 0.33 + 1 + 0.33 + 0.33 + 0.33 + 0.33 +0.33	| 3.97
| 4	| Therefore, Peter stayed with her at the hospital for 3 days without leaving |	1 + 0.67 + 0.67 + 0.33 + 0.33 + 0.33	| 3.33

From the sum of the weighted frequencies of the words, we can deduce that the first sentence carries the most weight in the paragraph. Therefore, it can give the best representative summary of what the paragraph is about.

Furthermore, if the first sentence is combined with the third sentence, which is the second-most weighty sentence in the paragraph, a better summary can be generated.

## Handy Links

- https://blog.floydhub.com/gentle-introduction-to-text-summarization-in-machine-learning/