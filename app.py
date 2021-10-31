# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
## https://arxiv.org/abs/1406.1078
## EuiYul Song (20214426)

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
from vocab import Vocab
from encoder import Encoder
from decoder import Decoder
import logging
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)


def preprocess() -> (Vocab, Vocab, list):
    file = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')

    translation_pair = [[re.sub(r"[^a-zA-Z.!?]+", r" ", re.sub(r"([.!?])", r" \1", ''.join(
        c for c in unicodedata.normalize('NFD', s.lower().strip())
        if unicodedata.category(c) != 'Mn'
    ))) for s in line.split('\t')] for line in file]

    translation_pair = [list(reversed(p)) for p in translation_pair]
    translation_input = Vocab()
    translation_output = Vocab()
    translation_pair = [pair for pair in translation_pair if (len(pair[0].split(' ')) < 10 and
                                                              len(pair[1].split(' ')) < 10 and
                                                              pair[1].startswith((
                                                                  "i am ", "he is", "she is", "you are", "we are",
                                                                  "they are", "i m ", "he s ", "she s ", "you re ",
                                                                  "we re ",
                                                                  "they re "
                                                              )
                                                              ))]

    for pair in translation_pair:
        translation_input.update(pair[0])
        translation_output.update(pair[1])
    logging.debug(f"A random pair from translation pair: {random.choice(translation_pair)}")

    return translation_input, translation_output, translation_pair


def tensor_sentence(vocab: Vocab, sentence: str) -> torch.tensor:
    logging.debug(f"vocab.word2idx: {vocab.word2idx}, sentence: {sentence.split(' ')}")
    idx = [vocab.word2idx[word] for word in sentence.split(' ') if (word in vocab.word2idx)]
    idx.append(1)
    return torch.tensor(idx, dtype=torch.long, device=device).view(-1, 1)


def query(encoder: Encoder, decoder: Decoder, num: int, input: str, inputs: Vocab,
          outputs: Vocab) -> str:
    with torch.no_grad():
        input_tensor = tensor_sentence(inputs, input)
        input_length = input_tensor.size()[0]
        encoder_hidden = torch.zeros(1, 1, encoder.hidden, device=device)

        encoder_outputs = torch.zeros(num, encoder.hidden, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[0]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(num):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(outputs.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()
    result = ' '.join(decoded_words).encode().decode('utf-8').replace(" <EOS>", "")
    logging.info(f"Model Result: {result}")
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input = request.form['search']
        result = query(encoder, decoder, 10, input, translation_input, translation_output)
        print(result)
        return render_template('index.html', input = input, result=result, is_result=True)
    return render_template('index.html', input = "", result="", is_result=False)



if __name__ == '__main__':
    hidden = 256
    epochs = 70000
    lr = 0.01
    s = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logging.basicConfig(filename=f"log/app{s}.log", encoding='utf-8',
                        level=logging.DEBUG)
    logging.info("=" * 10 + "Initializing Project" + "=" * 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    translation_input, translation_output, pairs = preprocess()

    logging.info(f"Initializing Encoder and Decoder...")
    encoder = Encoder(translation_input.word_count, hidden).to(device)
    decoder = Decoder(hidden, translation_output.word_count).to(device)

    logging.info(f"Loading Objects...")
    encoder.load_state_dict(torch.load('checkpoint/encoder2021-10-31-04:03:50.pth'))
    decoder.load_state_dict(torch.load('checkpoint/decoder2021-10-31-04:03:50.pth'))

    app.run()
