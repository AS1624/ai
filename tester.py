import ai
import jax.numpy as np
import csv

#loading the dataset
inputs:list[list[int]] = []
outputs:list[list[int]] = []
with open("MNIST_CSV/mnist_train.csv", "r", newline="\n") as data:
    reader = csv.reader(data)

    for row in reader:
        outputs.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        outputs[-1][int(row[0])] = 1
        inputs.append([int(x) for x in row])
        (inputs[-1]).pop(0)
        # if len(inputs) % 2 == 0:
        #     inputs.pop()
        #     outputs.pop()
print("loaded data")

test = ai.ai([784, 28, 10])

test.train(inputs, outputs, 1, 0.1)

inputs:list[list[int]] = []
outputs:list[list[int]] = []
with open("MNIST_CSV/mnist_test.csv", "r", newline="\n") as data:
    reader = csv.reader(data)

    for row in reader:
        outputs.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        outputs[-1][int(row[0])] = 1
        inputs.append([int(x) for x in row])
        (inputs[-1]).pop(0)

print(test.test(inputs, outputs))
