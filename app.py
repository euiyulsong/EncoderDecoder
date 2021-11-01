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


def preprocess(dir: str) -> (Vocab, Vocab, list):
    file = open(dir, encoding='utf-8').read().strip().split('\n')
    translation_pair = [[re.sub(r"[^a-zA-Z.!?]+", r" ", re.sub(r"([.!?])", r" \1", ''.join(
        c for c in unicodedata.normalize('NFD', s.lower().strip())
        if unicodedata.category(c) != 'Mn'
    ))) for s in line.split('\t')] for line in file]

    translation_pair = [list(reversed(p)) for p in translation_pair]
    translation_pair = [pair for pair in translation_pair if (len(pair[0].split(' ')) < 7 and
                                                              len(pair[1].split(' ')) < 7)]

    translation_input = Vocab()
    translation_output = Vocab()
    for pair in translation_pair:
        translation_input.update(pair[0])
        translation_output.update(pair[1])

    return translation_input, translation_output, translation_pair


def sentencing(vocab: Vocab, sentence: str) -> torch.tensor:
    logging.debug(f"vocab.word2idx: {vocab.word2idx}, sentence: {sentence.split(' ')}")
    idx = [vocab.word2idx[word] for word in sentence.split(' ') if (word in vocab.word2idx)]
    idx.append(1)
    return torch.tensor(idx, dtype=torch.long, device=device).view(-1, 1)

@torch.no_grad()
def query(encoder: Encoder, decoder: Decoder, num: int, input: str, inputs: Vocab,
          outputs: Vocab) -> str:
    input_pair = sentencing(inputs, input)
    len_input = input_pair.size()[0]
    e_h = torch.zeros(1, 1, encoder.hidden, device=device)

    e_o = torch.zeros(num, encoder.hidden, device=device)

    for i in range(len_input):
        temp, e_h = encoder(input_pair[i], e_h)
        e_o[i] += temp[0, 0]

    d_i = torch.tensor([[0]], device=device)

    d_h = e_h

    result = []

    for j in range(num):
        d_o, d_h = decoder(
            d_i, d_h)
        _, i = d_o.data.topk(1)
        if i.item() == 1:
            break
        result.append(outputs.idx2word[i.item()])

        d_i = i.squeeze().detach()
    result = ' '.join(result).encode().decode('utf-8')
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

    s = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logging.basicConfig(filename=f"log/app{s}.log", encoding='utf-8',
                        level=logging.DEBUG)
    logging.info("=" * 10 + "Initializing Project" + "=" * 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    translation_input, translation_output, pairs = preprocess("data/eng-fra-train.txt")


    logging.info(f"Initializing Encoder and Decoder...")
    encoder = Encoder(translation_input.word_count, hidden).to(device)
    decoder = Decoder(hidden, translation_output.word_count).to(device)

    logging.info(f"Loading Objects...")
    encoder.load_state_dict(torch.load('checkpoint/encoder2021-11-01-18:01:11.pth', map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load('checkpoint/decoder2021-11-01-18:01:11.pth', map_location=torch.device('cpu')))


    app.run()
