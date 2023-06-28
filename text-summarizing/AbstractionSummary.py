import pandas as pd
import numpy as np 
import tensorflow as tf

encoder_maxlen = 400
decoder_maxlen = 75
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def main():
    url = 'dataset/inshorts-clean-data.csv'
    news = pd.read_csv(url, header=0)

    # preprocessing
    news = news[["Headline", "Short"]]
    document = news[["Short"]]
    summary = news[["Headline"]]
    print(document.iloc[30])
    print(summary.iloc[30])
    # for recognizing the start and end of target sequences, we pad them with "<go>" and "<stop>"
    summary = summary.apply(lambda x: '<go> ' + x + ' <stop>')
    # fit tokenizer to filter punctuation marks, lower case the text, gets a vocabulary dict (frequency of occurence)
    # since < and > from default tokens cannot be removed
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    oov_token = '<unk>'

    document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
    summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)

    document_tokenizer.fit_on_texts(document)
    summary_tokenizer.fit_on_texts(summary)

    inputs = document_tokenizer.texts_to_sequences(document)
    targets = summary_tokenizer.texts_to_sequences(summary)

    # padding and truncating the sequences to a fixed length so that we can have a generalized input to the model
    # it means that we have a matrix of nxm
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

    print(inputs)
    print(targets)

    # use ts dataset api for faster computations (since df is now a tensor)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(dataset)


if __name__== "__main__":
    main()