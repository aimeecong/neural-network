
__author__ = 'Shu Cong'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains a deep neural network
    """

    def __init__(self, nn_input_dim, layer_dims, nn_output_dim, actFun_type='tanh', reg_lambda=0.01,seed=0,type):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: python array containing the dimensions of each layer
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        # self.nn_hidden_dim = nn_hidden_dim
        self.layer_dims=layer_dims
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.L = len(self.layer_dims)  #number of all the layers in the network including input and output
        self.type=type
        self.AL=AL

        # initialize the weights and biases in the network

        np.random.seed(seed)
        self.parameters={}
        for l in range(1,self.L):
            self.parameters['W'+str(l)]=np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) / np.sqrt(self.layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((1,  self.layer_dims[l]))

    def feedforward(self,X,activation):
        self.caches=[]
        A=X
        ##implement L-1 layer.add cache to caches list.
        A_prev = A
        L = len(self.parameters)//2
        for l in range(1,L):
            A,cache = Layer.feed_forward(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],activation)
            self.caches.append(cache)
        ##implement L-1 layer.add cache to caches list.

        self.AL, cache = Layer.feed_forward(A,self.parameters['W'+str(L)],self.parameters['b'+str(L)],activation='softmax')
        self.caches.append(cache)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        # self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        corect_logprobs = -np.log(self.AL[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X,  activation=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y,activation):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        # num_examples = len(X)
        # delta3 = self.probs
        # delta3[range(num_examples), y] -= 1
        # dW2 = np.dot(self.a1.T, delta3)
        # db2 = np.sum(delta3, axis=0, keepdims=True)
        # dW1 = np.dot(X.T, np.dot(delta3, self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type))  ##
        # db1 = np.sum(np.dot(delta3, self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type), axis=0)
        # return dW1, dW2, db1, db2
        grads = {}
        L = len(self.caches) #number of layers
        m = self.AL.shape[0]

        ##initializing back prop
        dZL = self.AL[range(m), y] - 1
        current_cache = self.caches[L-1]
        grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=Layer.linear_backward(dZL,current_cache)
        for l in reversed(range(L-1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp =Layer.backprop(grads["dA" + str(l + 1)], current_cache,activation)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp
        return grads

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, activation=self.actFun_type)
            # Backpropagation
            grads = self.backprop(X, lambda x: Layer.feedforward(x, type=self.actFun_type))

            # Add regularization terms (b1 and b2 don't have regularization terms)
            # dW2 += self.reg_lambda * self.W2
            # dW1 += self.reg_lambda * self.W1
            L=len(parameters)
            # Gradient descent parameter update
            # self.W1 += -epsilon * dW1
            # self.b1 += -epsilon * db1
            # self.W2 += -epsilon * dW2
            # self.b2 += -epsilon * db2
            for l in range(1,L+1):
                grads["dW" + str(l)]+=self.reg_lambda * self.parameters['W'+str(l)]
                self.parameters['W'+str(l)]=self.parameters['W'+str(l)]-epsilon * grads['W'+str(l)]
                self.parameters['b' + str(l)] = self.parameters['Wb' + str(l)] - epsilon * grads['b' + str(l)]
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)



class Layer(object):
    def __init__(self):
    pass


    def linear_forward(self,A,W,b):

        Z = np.dot(A,W)+b
        cache=(A,W,b)
        return Z,cache

    def feed_forward(self,A_prev,W,b,,activation):


        if activation == 'tanh':
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A = np.tanh(Z)
            activation_cache=(A,Z)

        elif activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = 1 / (1 + np.exp(-Z))
            activation_cache = (A, Z)

        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = np.maximum(0, Z)
            activation_cache = (A, Z)

        elif activation == 'softmax':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = np.exp(Z)/np.sum(np.exp(Z),axis=1,keepdims=True)
            activation_cache = (A, Z)

        cache = (linear_cache,activation_cache)
        return A, cache

    def linear_backward(self,dZ,cache):
        A_prev, W, b = cache
        dW = np.dot(A_prev.T,dZ)
        db = np.sum(dZ,axis=0,keepdims=True)
        dA_prev = np.dot(dZ,W.T)

        return dA_prev,dW,db

    def backprop(self, dA, cache,activation):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        linear_cache,activation_cache=cache
        Z = activation_cache
        A,W,b=linear_cache
        if activation =='sigmoid':
            dZ=dA * A * (1-A)
            dA_prev, dW, db =self.linear_backward(dZ,linear_cache)
        if activation == 'tanh':
            dZ = dA *  (1 - A**2)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        if activation == 'relu':
            dZ = dA * (1. * (Z > 0))
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db



def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=20, nn_output_dim=2, actFun_type='relu')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()