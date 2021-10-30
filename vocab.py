class Vocab:
    def __init__(self):
        self.word_freq = {}
        self.word2idx = {}
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.word_count = 2

    def update(self, sentence):
        for word in sentence.split(' '):
            if word in self.word2idx:
                self.word_freq[word] += 1
            else:
                self.word_freq[word] = 1
                self.word2idx[word] = self.word_count
                self.idx2word[self.word_count] = word
                self.word_count += 1