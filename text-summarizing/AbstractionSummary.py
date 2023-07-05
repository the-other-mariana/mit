import pandas as pd
import numpy as np 
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

# hyper-params
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
EPOCHS = 10

BUFFER_SIZE = 20000
BATCH_SIZE = 64

encoder_maxlen = 400
decoder_maxlen = 75

encoder_vocab_size, decoder_vocab_size = 76362, 29661

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
    return seq[:, tf.newaxis, tf.newaxis, :]

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

# two layer feed forward network which is used after almost every sublayer
# add & norm blocks join a residual connection and transforms it with a Normalization Layer
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

# d_model usually is the number of dimensions in each word embedding
# define mh attention as a custom tf layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # defining these as instances of Dense Layers with d_model as neurons in the layer
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # the call method gets called when the object is called, not initialized
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # applies the layer to the params: matmul between q and the weights in the layer (learned)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # splits result into h parts
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # applies attention to each h part
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concat the h parts again into its original shape
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # linear step again 
        output = self.dense(concat_attention)
            
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        # creates the layer and neurons
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # calls the model with the created layers and neurons
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # as much self attention layers as specified
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
    
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
        return x, attention_weights

# this class inherits from tf.keras.Model class
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

# the transformer paper also suggests training on a custom learning rate scheduler that helps faster convergence
# determines the size of the parameters update when optimizing
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Cast `step` to float
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def evaluate(input_document, document_tokenizer, summary_tokenizer):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def summarize(input_document, summary_tokenizer):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document

def load_model_for_inference(export_path, new_input):
    # load the saved model
    loaded_model = tf.saved_model.load(export_path)
    # Perform inference using the loaded model without retraining the whole model
    # You can call the model on new inputs
    # For example, assuming you have a new input sequence called `new_input`
    predictions, _ = loaded_model(new_input, training=False)
    # `predictions` will contain the model's predictions for the new input

    # You can also access individual layers of the loaded model
    # For example, to get the encoder layer
    encoder = loaded_model.encoder
    # Use the encoder layer for further processing ...


def main():

    url = 'dataset/news.xlsx'
    news = pd.read_excel(url)

    # preprocessing
    news.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
    document = news["Short"]
    summary = news["Headline"]
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

    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)

    print(inputs)
    print(targets)

    # use ts dataset api for faster computations (since df is now a tensor)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(dataset)

    # Create a custom learning rate schedule
    learning_rate = CustomSchedule(d_model)
    # Plot the learning rate schedule test
    plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.show()
    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # Initialize the loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    # Initialize the metric for evaluation
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # instantiate the model with the params and configs
    global transformer
    transformer = Transformer(
    num_layers, 
    d_model, 
    num_heads, 
    dff,
    encoder_vocab_size, 
    decoder_vocab_size, 
    pe_input=encoder_vocab_size, 
    pe_target=decoder_vocab_size,
    )
    
    checkpoint_path = "checkpoints"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, 
                True, 
                enc_padding_mask, 
                combined_mask, 
                dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)   
        # the optimizer (adam) updates the model's trainable variables  
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
    
        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(inp, tar)
        
            # 55k samples
            # we display 3 batch results -- 0th, middle and last one (approx)
            # 55k / 64 ~ 858; 858 / 2 = 429
            #print(batch)
            #if batch % 429 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, train_loss.result()))
        
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        
        print ('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # export (save) the trained model
    export_path = "saved_model"
    tf.saved_model.save(transformer, export_path)
    print("Model exported to:", export_path)

    summarize(
        "US-based private equity firm General Atlantic is in talks to invest about \
        $850 million to $950 million in Reliance Industries' digital unit Jio \
        Platforms, the Bloomberg reported. Saudi Arabia's $320 billion sovereign \
        wealth fund is reportedly also exploring a potential investment in the \
        Mukesh Ambani-led company. The 'Public Investment Fund' is looking to \
        acquire a minority stake in Jio Platforms."
    )

    # should print something similar to: 'reliance group buys stake in net profit for 250 bn'


if __name__== "__main__":
    main()