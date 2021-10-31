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
import sys



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
                                                                  "they are", "i m ", "he s ", "she s ", "you re ", "we re ",
                                                                  "they re "
                                                              )
                                                              ))]

    for pair in translation_pair:
        translation_input.update(pair[0])
        translation_output.update(pair[1])
    logging.debug(f"A random pair from translation pair: {random.choice(translation_pair)}")

    return translation_input, translation_output, translation_pair


def tensor_sentence(vocab: Vocab, sentence: str) -> torch.tensor:
    idx = [vocab.word2idx[word] for word in sentence.split(' ') if (word in vocab.word2idx)]
    idx.append(1)
    return torch.tensor(idx, dtype=torch.long, device=device).view(-1, 1)

def train(encoder: Encoder, decoder: Decoder, epochs: int, translation_input: Vocab, translation_output: Vocab,
          pairs: list) -> None:
    loss_sum = 0

    encoder_opt = torch.optim.SGD(encoder.parameters(), lr=0.01)
    decoder_opt = torch.optim.SGD(decoder.parameters(), lr=0.01)
    random_pair = random.choice(pairs)
    training_pairs = [(tensor_sentence(translation_input, random_pair[0]),
                       tensor_sentence(translation_output, random_pair[1]))
                      for i in range(epochs)]
    nll = torch.nn.NLLLoss()

    for idx in range(1, epochs + 1):
        training_pair = training_pairs[idx - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        tfr = 0.5

        encoder_hidden = torch.zeros(1, 1, encoder.hidden, device=device)

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(10, encoder.hidden, device=device)

        loss = 0

        for idx2 in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[idx2], encoder_hidden)
            encoder_outputs[idx2] = encoder_output[0, 0]

        decoder_input = torch.tensor([[0]], device=device)

        is_tf = (random.random() < tfr)

        if is_tf:
            for idx2 in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, encoder_hidden)
                loss += nll(decoder_output, target_tensor[idx2])
                decoder_input = target_tensor[idx2]

        else:
            for idx2 in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, encoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += nll(decoder_output, target_tensor[idx2])
                if decoder_input.item() == 1:
                    break

        loss.backward()

        encoder_opt.step()
        decoder_opt.step()
        loss = loss.item() / target_length
        loss_sum += loss

        if idx % 1000 == 0:
            loss_avg = loss_sum / 1000
            logging.info(f"Epoch {idx}: {loss_avg} loss")
            loss_sum = 0


def evaluate(encoder: Encoder, decoder: Decoder, num: int, translation_pairs: list, inputs: Vocab,
             outputs: Vocab) -> None:
    bleu_sum = 0
    for i in range(num):
        pair = random.choice(translation_pairs)
        logging.info(f"French: { pair[0] }, English: { pair[1] }")

        with torch.no_grad():
            input_tensor = tensor_sentence(inputs, pair[0])
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
        logging.info(f"Model Result: { result }")



if __name__ == '__main__':
    hidden = 256
    epochs = 70000
    lr = 0.01
    s = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    arg = sys.argv[1]
    logging.basicConfig(filename=f"log/{s}.log", encoding='utf-8',
                        level=logging.DEBUG)
    logging.info("=" * 10 + "Initializing Project" + "=" * 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    logging.info(f"Start Preprocessing...")
    translation_input: Vocab
    translation_input, translation_output, pairs = preprocess()

    logging.info(f"Initializing Encoder and Decoder...")
    encoder = Encoder(translation_input.word_count, hidden).to(device)
    decoder = Decoder(hidden, translation_output.word_count).to(device)
    logging.info(f"Start Training...")
    train(encoder, decoder, 80000, translation_input, translation_output, pairs)

    logging.info(f"Saving Objects...")

    torch.save(f'checkpoint/encoder{s}.pth')
    torch.save(f'checkpoint/decoder{s}pth')

    logging.info(f"translation_input: { translation_input}")
    logging.info(f"translation_output: { translation_output}")

    logging.info("Start Evaluating...")
    evaluate(encoder, decoder, 10, pairs, translation_input, translation_output)



