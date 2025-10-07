# starter code for a2

Add the corresponding (one) line under the ``[to fill]`` in ``def forward()`` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

One example on running the code:

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 --train_data ./training.json --val_data ./validation.json``


**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 --train_data training.json --val_data validation.json``

**Added Lines to ffnn.py**
Lines added in def forward on ffnn.py

We added a hiddenLayer to with self.W1 which is a linear layer that transforms the input vector to a vector of dimenson h which is the hidden dimension. Then the activation ReLU function apllies to linear transformation meaning the line computes the hidden layer

The outputLayer contains the self.W2 linear layer that transforms hiddenLayer to a vector of self.ouput_dim of 5 becuase of the 5 different ratings

The predict_vector is the softmax function applied to the outputLayer to convert raw scores into log probabilities for computational benefits with numerical stability.

The return predictied vector returns the log probability dist over every class

Libraries/Tools: Pytorch torch and torch.nn function to use the nn.Linear (for self.W1 and self.W2 for performing linear transformation), nn.ReLU (for self.activation to apply non-linearity), and nn.LogSoftmax(for self.softmax to convert raw scores into log probabilities) for numerical for the architecture of the neural network. Core Python functions are mainly for function calls and constructs.

**Added Lines to rnn.py**

Lines added to def forward

The hidden self.rnn(inputs) passes the input sequence through RNN layer. The RNN processes the sequence to return:
- The output of each time step
- The final hidden state of RNN

outputLayer appplies a linear transformation to the final hidden state to map the hidden state to a vector of size 5 (output classes, star ratings)

outputLayer.sum adds the output across the specfic dimesion which may aggregrate the outputs if RNN produces many hidden states in a batch

predicted_vector applies softmax to output layer and convert to a probability distribution over 5 classes. Results vectro probabilities where each value represents class likelihood

Libraries/Tools: Pytorch torch and torch.nn function to use the nn.Linear (for self.W1 and self.W2 for performing linear transformation), nn.ReLU (for self.activation to apply non-linearity), and nn.LogSoftmax(for self.softmax to convert raw scores into log probabilities) for numerical for the architecture of the neural network. Core Python functions are mainly for function calls and constructs.