import math
import numpy as np

def sigmoid(x: float) -> float:
    if isinstance(x, list):
        x = np.array(x)

    if isinstance(x, np.ndarray):
        output = []
        for elem in x:
            output.append(sigmoid(elem))
        return np.array(output)
    return 1 / ( 1 + math.exp(-x))
    # 1 / ( 1 + e ^ (-x) )

def sigmoid_deriv(x: float) -> float:
    if isinstance(x, list):
        x = np.array(x)
    return sigmoid(x) * ( 1 - sigmoid(x))

class ai:

    def __init__(self, layers: list[int]):
        self.network:list[list[np.ndarray]] = []
        for i in range(len(layers) - 1):
            self.network.append([])
            for _ in range(layers[i + 1]):
                self.network[i].append(self.randomWeights(layers[i]))

    def feedforward(self, inputs:list[float]) -> list[float]:
        self.a:list[list[float]] = [inputs.copy()]
        self.z:list[list[float]] = [inputs.copy()]

        self.z[0].append(1)
        self.a[0].append(1)
        for i in range(len(self.network)):
            self.a.append([])
            self.z.append([])
            for j in range(len(self.network[i])):
                computed = self.compute(
                    self.a[i], 
                    self.network[i][j]
                )
                self.z[i + 1].append(computed)
                self.a[i + 1].append(
                    sigmoid(computed)
                )
            self.z[i + 1].append(1)
            self.a[i + 1].append(1)

        
        return self.a[-1]
    
    def backpropagate(self, expected:list[float], learning_rate:float):
        error = (
            np.array([self.a[-1][x] for x in range(len(self.a[-1]) -1)])
            - np.array(expected)
        )
        for i in range(len(self.network) - 1, -1, -1):
            if i > 0:
                newErrors = np.zeros((len(self.network[i - 1])))
            for j in range(len(self.network[i])):
                self.network[i][j] -= (
                    np.array(self.a[i])
                    * sigmoid_deriv(self.z[i + 1][j]) 
                    * error[j] 
                    * learning_rate
                )
                if i > 0:
                    for k in range(len(self.network[i - 1])):
                        newErrors[k] += (
                            self.network[i][j][k] 
                            * sigmoid_deriv(self.z[i + 1][j])
                            * error[j]
                            * 1000
                        )
            if i > 0:
                error = newErrors.copy()
         
    
    def compute(self, inputs:list[float], weights:np.ndarray) -> float:
        newIn = np.array(inputs)

        return float(
            (
                np.atleast_2d(newIn) 
                @ np.atleast_2d(weights).T
            )[0][0]
        )
    
    def train(
    self, 
    inputs:list[list[float]], 
    outputs:list[list[float]], 
    epochs:float, 
    learning_rate:float):
        for epoch in range(1, epochs + 1):
            print("epoch {:5d} / {:5d}".format( epoch, epochs))
            for i in range(len(inputs)):
                if i % math.ceil(len(inputs) * 0.001) == 0:
                    print("{:.2f}%".format(i / len(inputs) * 100))
                self.feedforward(inputs[i])
                self.backpropagate(outputs[i], learning_rate)
            
    def test(self, inputs:list[list[float]], outputs:list[list[float]]) -> float:
        correct = 0
        for i in range(len(inputs)):
            output = self.feedforward(inputs[i])
            print(["{:0.2f}".format(x) for x in output], "\n", outputs[i], "\n\n\n")
            max = 0
            maxIdx = 0 
            for j in range(len(output) - 1):
                if output[j] > max:
                    max = output[j]
                    maxIdx = j

            if maxIdx == outputs[i].index(1):
                correct += 1
        
        return correct / len(inputs)


    def randomWeights(self, numInputs:int):
        return np.random.rand(numInputs + 1)
if __name__ == "__main__":    
    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    outputs = [
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
    ]

    test = ai([2, 3, 2])

    test.train(inputs, outputs, 10000, 0.5)

    print("\n\n")

    print(test.test(inputs, outputs))