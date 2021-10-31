from ignite.metrics.nlp import Bleu

m = Bleu(ngram=2, smooth="smooth1")

y_pred = "i m capable of making my own decisions ."
y = "i am capable of doing my duty ."

m.update((y_pred.split(), [y.split()]))

print(m.compute().item())
