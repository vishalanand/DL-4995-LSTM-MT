from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu

import argparse, numpy as np, os, pickle

def call_plot(loss, val_loss):    
    epochs = range(1,len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss (' + str(str(format(loss[-1],'.5f'))+')'))
    plt.plot(epochs, val_loss, 'g', label='Validation loss (' + str(str(format(val_loss[-1],'.5f'))+')'))
    plt.title('va2361: Machine Translation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def encoder_input_data_var(data_path, num_samples):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    
    return encoder_input_data

def text_var(data_path, num_samples):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        #input_text = input_text + "1"
        #print(input_text)
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    
    return num_encoder_tokens, num_decoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, target_texts, max_decoder_seq_length
    
def call_score_evaluation(seq_l, encoder_model, num_decoder_tokens, target_token_index, decoder_model, reverse_target_char_index, max_decoder_seq_length, target_texts, encoder_input_data_test):
    bleu_scores = []
    for seq_index in range(seq_l):
        input_seq = encoder_input_data_test[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, 
                                           num_decoder_tokens, target_token_index, 
                                           decoder_model, reverse_target_char_index, max_decoder_seq_length)
        text1 = ' '.join(decoded_sentence[:-1].split())
        text2 = target_texts[seq_index][1:][:-1]
        bleu_score_val = sentence_bleu([text_to_word_sequence(text1)], text_to_word_sequence(text2))
        print(text1, ":\t", text2, ",\tBLEU:", bleu_score_val)
        bleu_scores.append(bleu_score_val)

    return bleu_scores

def define_model(batch_size, epochs, latent_dim, num_samples, data_path):
    
    num_encoder_tokens, num_decoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, target_texts, max_decoder_seq_length = text_var(data_path, num_samples)

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    
    #encoder_input_data_train, encoder_input_data_test, decoder_input_data_train, decoder_input_data_test, decoder_target_data_train, decoder_target_data_test = train_test_split(encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.2, random_state=1)

    a_train, a_test = train_test_split(range(len(decoder_target_data)), test_size=0.2, random_state=1)
    #print(a_train, a_test)
    #print(len(a_train), len(a_test))
    encoder_input_data_train = encoder_input_data[a_train]
    encoder_input_data_test = encoder_input_data[a_test]
    decoder_input_data_train = decoder_input_data[a_train]
    decoder_input_data_test = decoder_input_data[a_test]
    decoder_target_data_train = decoder_input_data[a_train]
    decoder_target_data_test = decoder_target_data[a_test]

    #hist = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    hist = model.fit([encoder_input_data_train, decoder_input_data_train], decoder_target_data_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    test_accuracy = model.evaluate([encoder_input_data_test, decoder_input_data_test], decoder_target_data_test, verbose=0)
    print("Test Accuracy:", test_accuracy)
    model.save('models/model.mod')

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


    bleu_scores = call_score_evaluation(len(encoder_input_data_test), encoder_model, num_decoder_tokens, target_token_index, decoder_model, reverse_target_char_index, max_decoder_seq_length, target_texts, encoder_input_data_test)

    print(bleu_scores)
    
    return hist.history['loss'], hist.history['val_loss'], target_texts, encoder_model, num_decoder_tokens, target_token_index, decoder_model, reverse_target_char_index, max_decoder_seq_length, test_accuracy, bleu_scores

def decode_sequence(input_seq, encoder_model, num_decoder_tokens, target_token_index, decoder_model, reverse_target_char_index, max_decoder_seq_length):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_path', default = "data/deu-eng/deu.txt")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    latent_dim = args.latent_dim
    num_samples = args.num_samples
    data_path = args.data_path

    loss, val_loss, target_texts, encoder_model, num_decoder_tokens, target_token_index, decoder_model, reverse_target_char_index, max_decoder_seq_length, test_accuracy, bleu_scores = define_model(batch_size, epochs, latent_dim, num_samples, data_path)

    with open('models/objs.pkl', 'wb') as f:
        pickle.dump([loss, val_loss, target_texts, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length, bleu_scores], f)

    encoder_model.save("models/encoder.mod")
    decoder_model.save("models/decoder.mod")

    [loss, val_loss, target_texts, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length, bleu_scores] = pickle.load(open( "models/objs.pkl", "rb" ))
    
    '''
    encoder_input_data = encoder_input_data_var(data_path, num_samples)
    for seq_index in range(10):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, 
                                           num_decoder_tokens, target_token_index, 
                                           decoder_model, reverse_target_char_index, max_decoder_seq_length)
        #text1 = input_texts[seq_index]
        text1 = ' '.join(decoded_sentence[:-1].split())
        text2 = target_texts[seq_index][1:][:-1]
        bleu_score_val = sentence_bleu([text_to_word_sequence(text1)], text_to_word_sequence(text2))
        print(text1, ":\t", text2, ",\tBLEU:", bleu_score_val)
    '''

#python -W ignore enc.py --data_path="data/deu-eng/deu.txt" --batch_size=64 --latent_dim=256 --epochs=20 --num_samples=10000
#python -W ignore enc.py --data_path="data/deu-eng/deu.txt" --batch_size=64 --latent_dim=256 --epochs=10 --num_samples=159204
#python -W ignore enc.py --data_path="data/deu-eng/deu.txt" --batch_size=64 --latent_dim=256 --epochs=20 --num_samples=100
#python -W ignore enc.py --data_path="data/deu-eng/deu.txt" --batch_size=64 --latent_dim=256 --epochs=20 --num_samples=159204 > a.txt

