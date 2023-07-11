import pandas as pd
import numpy as np 
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import seaborn as sns
import random
from GridPlot import GridPlot

from AbstractionSummary import create_masks

# hyper-params
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
EPOCHS = 20

BUFFER_SIZE = 20000
BATCH_SIZE = 64

encoder_maxlen = 400
decoder_maxlen = 75

encoder_vocab_size, decoder_vocab_size = 76362, 29661
attention_weights_list = []
probabilities_list = []

def evaluate(input_document, document_tokenizer, summary_tokenizer):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)
    print('enc shape:', encoder_input)

    #decoder_input = [summary_tokenizer.word_index["<go>"]] * (decoder_maxlen - 1)
    #decoder_input = [0] * (decoder_maxlen - 2) + [summary_tokenizer.word_index["<go>"]]
    decoder_input = [summary_tokenizer.word_index["<go>"]] + [0] * (decoder_maxlen - 2)

    decoder_input = tf.expand_dims(decoder_input, 0)
    print(decoder_input)
    #decoder_input = tf.pad(decoder_input, [[0, 0], [0, decoder_maxlen - 1 - len(decoder_input[0])]], constant_values=0)

    output = decoder_input
    print('dec shape:', output)
    
    for i in range(decoder_maxlen):
        print('------')
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        print(output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        attention_weights_list.append(attention_weights['decoder_layer4_block2'][0])
        step_probabilities = predictions[:, i, :]
        print('PROB', step_probabilities)
        probabilities_list.append(step_probabilities)

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        print('id', predicted_id)
        print('att', attention_weights['decoder_layer4_block2'][0])

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = output[:, -73:]
        output = tf.concat([output, predicted_id], axis=-1)
        #output = tf.concat([output[:, 1:], predicted_id], axis=-1)
        print('predict', output)

    return tf.squeeze(output, axis=0), attention_weights


def summarize(input_document, document_tokenizer, summary_tokenizer):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document, document_tokenizer, summary_tokenizer)[0].numpy()
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

    inputs = document_tokenizer.texts_to_sequences(document)
    targets = summary_tokenizer.texts_to_sequences(summary)

    # padding and truncating the sequences to a fixed length so that we can have a generalized input to the model
    # it means that we have a matrix of nxm
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)

    # use ts dataset api for faster computations (since df is now a tensor)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # instantiate the model with the params and configs
    global transformer
    transformer = tf.saved_model.load('saved_model')

    s = summarize(
        "US-based private equity firm General Atlantic is in talks to invest about \
        $850 million to $950 million in Reliance Industries' digital unit Jio \
        Platforms, the Bloomberg reported. Saudi Arabia's $320 billion sovereign \
        wealth fund is reportedly also exploring a potential investment in the \
        Mukesh Ambani-led company. The 'Public Investment Fund' is looking to \
        acquire a minority stake in Jio Platforms.",
        document_tokenizer,
        summary_tokenizer
    )

    print(s)

    print(attention_weights_list[0].shape)

    word_labels_encoder = list(document_tokenizer.index_word.values())
    word_labels_decoder = list(summary_tokenizer.index_word.values())

    submatrix_size = 10

    gp = GridPlot(len(attention_weights_list), num_heads, 'imgs/', (30, 15))

    for i in range(len(attention_weights_list)):
        for h in range(num_heads):
        
            fig, ax = plt.subplots(figsize=(8, 4))
            data = attention_weights_list[i]
            #im = ax.imshow(data[h,:62,:62], cmap='hot', extent=[0, 62, 62, 0])

            data = data[h,:62,:62]

            max_attention_index = np.argmax(data, axis=1)
            timestep = h
            word_index = max_attention_index[timestep]
            word = summary_tokenizer.index_word[word_index]

            ## random stuff
            num_rows, num_cols = data.shape[0], data.shape[1]
            chosen_position = np.unravel_index(word_index, (num_rows, num_cols))
            
            
            start_row = max(chosen_position[0] - submatrix_size // 2, 0)
            end_row = min(start_row + submatrix_size, num_rows)
            start_col = max(chosen_position[1] - submatrix_size // 2, 0)
            end_col = min(start_col + submatrix_size, num_cols)
            submatrix = data[start_row:end_row, start_col:end_col]
            max_position = np.unravel_index(np.argmax(submatrix), submatrix.shape)
            highlight_matrix = np.zeros((submatrix_size, submatrix_size))
            highlight_matrix[max_position] = 1

            x_labels = [summary_tokenizer.index_word[start_col+k] if k != 0 else '<unk>' for k in range(submatrix_size)]
            y_labels = [summary_tokenizer.index_word[start_row+k] if k != 0 else '<unk>' for k in range(submatrix_size)]

            im = ax.imshow(submatrix, cmap='gray')
            ax.imshow(highlight_matrix, cmap='cool', alpha=0.3)
            plt.colorbar(im)
            ax.set_xticks(range(submatrix_size), labels=x_labels, rotation=90)
            ax.set_yticks(range(submatrix_size), labels=y_labels)
            # Add labels, title, and colorbar
            ax.set_xlabel('Encoder timestep')
            #ax.set_xticks(len(word_labels))
            #ax.set_xticklabels(word_labels)
            
            ax.set_ylabel('Decoder timestep')
            ax.set_title(f'Attention Heatmap - Step {i+1} Head {h+1}')

            # Save the heatmap as an image
            plt.tight_layout()
            plt.savefig(f'imgs/heatmap_step_{i+1}_head{h+1}.png', dpi=500)
            plt.close()

            gp.plot_cell(i, h, submatrix, 
                         highlight_matrix, True, 'gray', f'Step {i+1} Head {h+1}',
                         'Encoder Timestep', 'Decoder Timestep', x_labels,
                         y_labels, True)
    gp.save_plot('heatmaps.png')

    for i in range(len(probabilities_list)):
        # Plot the heatmap
        step_probabilities = probabilities_list[i]
        plt.figure()
        ax = sns.heatmap(step_probabilities.numpy().reshape(-1, 1), cmap='Blues', annot=False, cbar=False, yticklabels=False)
        #ax.set_xticks(np.arange(num_words))
        #ax.set_xticklabels(document_tokenizer.index_word.values(), rotation=90)
        plt.title(f'Step {i+1} Probabilities Heatmap')
        plt.xlabel('Word')
        plt.ylabel('Batch')
        plt.savefig(f'imgs/prob_heatmap_step_{i+1}_head{h+1}.png', dpi=500)
        plt.close()

if __name__ == '__main__':
    main()