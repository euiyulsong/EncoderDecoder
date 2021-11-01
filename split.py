import random

file = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')
random.shuffle(file)
test_pair = file[int(0.9*(len(file))):]
print(len(test_pair))
train_pair = file[:int(0.9*(len(file)))]
print(len(train_pair))
with open('data/eng-fra-train.txt', 'w') as output:
    for i in train_pair:
        output.write(i + "\n")

with open('data/eng-fra-test.txt', 'w') as output:
    for i in test_pair:
        output.write(i + "\n")