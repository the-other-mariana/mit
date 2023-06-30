import pandas as pd
import numpy as np 
import tensorflow as tf

encoder_maxlen = 400
decoder_maxlen = 75
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

# positional encoding ensures the attention model preserves sequence in words, like a RNN
def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i + 1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    req[:, tf.newaxis, tf.newaxis, :]

# look ahead mask is responsible for ignoring the words that occur after a given current word
# in the target sequence
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

# basis for attention computation
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

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

    # this maps an index to each word and thus makes a list of words appear as an integer vector
    # word1 = 0 word2 = 1 and so on
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