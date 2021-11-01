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
from ignite.metrics.nlp import Bleu


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


def train(encoder: Encoder, decoder: Decoder, epochs: int, translation_input: Vocab, translation_output: Vocab,
          pairs: list) -> None:
    loss_sum = 0

    pairings = [pairing(random.choice(pairs), translation_input, translation_output) for i in range(epochs)]

    nll = torch.nn.NLLLoss()

    encoder_opt = torch.optim.SGD(encoder.parameters(), lr=0.01)
    decoder_opt = torch.optim.SGD(decoder.parameters(), lr=0.01)

    for idx in range(1, epochs + 1):
        pair = pairings[idx - 1]
        input_pair = pair[0]
        output_pair = pair[1]

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss = 0

        e_h, e_o, d_i, e_h, d_h = get_encoder_output(input_pair, encoder)

        for idx2 in range(output_pair.size(0)):
            d_o, d_h = decoder(d_i, d_h)
            _, i = d_o.topk(1)
            d_i = i.squeeze().detach()

            loss += nll(d_o, output_pair[idx2])
            if d_i.item() == 1:
                break

        loss.backward()

        encoder_opt.step()
        decoder_opt.step()
        loss = loss.item() / output_pair.size(0)
        loss_sum += loss

        if idx % 1000 == 0:
            print(f"Epoch {idx}: {loss_sum / 1000} loss")
            loss_sum = 0


def get_encoder_output(input_pair: list, encoder: Encoder):
    e_h = torch.zeros(1, 1, encoder.hidden, device=device)
    e_o = torch.zeros(7, encoder.hidden, device=device)

    for i in range(input_pair.size(0)):
        temp, e_h = encoder(input_pair[i], e_h)
        e_o[i] = temp[0, 0]
    d_i = torch.tensor([[0]], device=device)
    d_h = e_h
    return e_h, e_o, d_i, e_h, d_h


@torch.no_grad()
def evaluate(encoder: Encoder, decoder: Decoder, translation_pairs: list, inputs: Vocab,
             outputs: Vocab) -> None:
    bleu1_sum = bleu2_sum = bleu3_sum = bleu4_sum = 0
    for pair in translation_pairs:
        input_pair = sentencing(inputs, pair[0])

        e_h, e_o, d_i, e_h, d_h = get_encoder_output(input_pair, encoder)

        results = []

        for idx in range(7):
            d_o, d_h = decoder(d_i, d_h)
            _, i = d_o.data.topk(1)
            if i.item() == 1:
                break
            results.append(outputs.idx2word[i.item()])
            d_i = i.squeeze().detach()

        result = ' '.join(results).encode().decode('utf-8')
        m1 = Bleu(ngram=1, smooth="smooth1")
        m2 = Bleu(ngram=2, smooth="smooth1")
        m3 = Bleu(ngram=3, smooth="smooth1")
        m4 = Bleu(ngram=4, smooth="smooth1")
        m1.update((pair[1].split(), [result.split()]))
        m2.update((pair[1].split(), [result.split()]))
        m3.update((pair[1].split(), [result.split()]))
        m4.update((pair[1].split(), [result.split()]))
        bleu1_sum += (m1.compute().item())
        bleu2_sum += (m2.compute().item())
        bleu3_sum += (m3.compute().item())
        bleu4_sum += (m4.compute().item())
        logging.info(f"{pair[1]} == {result}")

    print(f"BLEU1: {bleu1_sum / len(translation_pairs)}")
    print(f"BLEU2: {bleu2_sum / len(translation_pairs)}")
    print(f"BLEU3: {bleu3_sum / len(translation_pairs)}")
    print(f"BLEU4: {bleu4_sum / len(translation_pairs)}")


def sentencing(vocab: Vocab, sentence: str) -> torch.tensor:
    idx = [vocab.word2idx[word] for word in sentence.split(' ') if (word in vocab.word2idx)]
    idx.append(1)
    return torch.tensor(idx, dtype=torch.long, device=device).view(-1, 1)


def pairing(pair: list, translation_input: Vocab, translation_output: Vocab):
    return (sentencing(translation_input, pair[0]), sentencing(translation_output, pair[1]))


if __name__ == '__main__':
    hidden = 256
    epochs = 80000
    lr = 0.01
    s = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logging.basicConfig(filename=f"log/{s}.log", encoding='utf-8',
                        level=logging.DEBUG)
    logging.info("=" * 10 + "Initializing Project" + "=" * 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    logging.info(f"Start Preprocessing...")
    translation_input, translation_output, pairs = preprocess("data/eng-fra-train.txt")
    _, _, test_pairs = preprocess("data/eng-fra-test.txt")
    logging.info(f"Initializing Encoder and Decoder...")
    encoder = Encoder(translation_input.word_count, hidden).to(device)
    decoder = Decoder(hidden, translation_output.word_count).to(device)
    logging.info(f"Start Training...")
    """
    train(encoder, decoder, epochs, translation_input, translation_output, pairs)

    logging.info(f"Saving Objects...")

    torch.save(encoder.state_dict(), f'checkpoint/encoder{s}.pth')
    torch.save(decoder.state_dict(), f'checkpoint/decoder{s}.pth')
    """
    logging.info(f"Loading Objects...")
    encoder.load_state_dict(torch.load('checkpoint/encoder2021-11-01-18:01:11.pth', map_location=torch.device('cpu') ))
    decoder.load_state_dict(torch.load('checkpoint/decoder2021-11-01-18:01:11.pth', map_location=torch.device('cpu')))

    logging.info("Start Evaluating Training...")
    evaluate(encoder, decoder, pairs, translation_input, translation_output)
    logging.info("Start Evaluating Testing...")
    evaluate(encoder, decoder, test_pairs, translation_input, translation_output)
