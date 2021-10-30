from ignite.metrics.nlp import Bleu

m = Bleu(ngram=4, smooth="smooth1")

y_pred = "apple"
y = "apple"

m.update((y_pred.split(), y.split()))

print(m.compute().item())
