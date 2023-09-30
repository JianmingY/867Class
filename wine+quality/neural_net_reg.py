import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

class NeuralNet_30_Neurons():
    def __init__(self, N_layer1 = 16, N_layer2 = 8, N_layer3 = 4, N_layer4 = 2):  # defined total 30 hidden neurons
        super(NeuralNet_30_Neurons,self).__init__()

        self.w1 = np.random.random((11, N_layer1)) / np.square(11/2)
        self.w2 = np.random.random((N_layer1, N_layer2)) / np.square(N_layer1/2)
        self.w3 = np.random.random((N_layer2, N_layer3)) / np.square(N_layer2/2)
        self.w4 = np.random.random((N_layer3, N_layer4)) / np.square(N_layer3/2)
        self.w5 = np.random.random((N_layer4, 1)) / np.square(N_layer4/2)
        self.bias = 1


    # Listing the matrix multipliaction between layers plus bias for each feature
    def eleven_to_sixteen(self, X):
        return np.dot(X,self.w1) + self.bias

    def sixteen_to_eight(self, X):
        return np.dot(X,self.w2) + self.bias

    def eight_to_four(self, X):
        return np.dot(X, self.w3) + self.bias

    def four_to_two(self, X):
        return np.dot(X, self.w4) + self.bias

    def two_to_output(self, X):
        return np.dot(X, self.w5) + self.bias

    def relu_activation(self, X):
        return np.maximum(0, X)

    def RMSE(self, Squared_error, y_test):
        '''
        Root mean squared error - another measure of distance between true label and prediction
        :param Squared_error: squared difference between each pair of true label and prediction
        :param y_test: the test data set
        :return: the rooted mean of all squared errors
        '''
        return (Squared_error/len(y_test)) ** 0.5
    def backward(self, X, y_true, learning_rate):
        '''
        Make predictions on train set in forward pass while training the model using backward propagation
        :param X: 11 training features
        :param y_true: training labels
        :param learning_rate: the size of step when adjusting parameters
        :return: prediction of y/wine quality
        '''
        # Forward pass: go through 4 layers of total 30 hidden neurons with relu-activation in between
        # Using softmax activation to perform final classification
        z1 = self.eleven_to_sixteen(X)
        a1 = self.relu_activation(z1)
        z2 = self.sixteen_to_eight(a1)
        a2 = self.relu_activation(z2)
        z3 = self.eight_to_four(a2)
        a3 = self.relu_activation(z3)
        z4 = self.four_to_two(a3)
        a4 = self.relu_activation(z4)
        z5 = self.two_to_output(a4)
        y_pred = self.relu_activation(z5)


        # Backpropagation: 1. backward propagate the RMSE's derivative
        # 2.backward propagate activation and hidden neurons to get the change
        delta5 = (y_pred - y_true) * (y_pred > 0)
        delta4 = np.dot(self.w5, delta5) * (a4 > 0)
        delta3 = np.dot(self.w4, delta4) * (a3 > 0)
        delta2 = np.dot(self.w3, delta3) * (a2 > 0)
        delta1 = np.dot(self.w2, delta2) * (a1 > 0)

        # Update weights and biases by subtracting the gradient from weights
        self.w5 -= learning_rate * np.dot(a4[:, np.newaxis],delta5[np.newaxis,:])
        self.w4 -= learning_rate * np.dot(a3[:, np.newaxis],delta4[np.newaxis,:])
        self.w3 -= learning_rate * np.dot(a2[:, np.newaxis],delta3[np.newaxis,:])
        self.w2 -= learning_rate * np.dot(a1[:, np.newaxis],delta2[np.newaxis,:])
        self.w1 -= learning_rate * np.dot(np.array(X)[:, np.newaxis], delta1[np.newaxis,:])

        return y_pred

    def forward(self, X):
        '''
        forwar pass to make predictions
        :param X: test set
        :return: prediction of y
        '''
        z1 = self.eleven_to_sixteen(X)
        a1 = self.relu_activation(z1)
        z2 = self.sixteen_to_eight(a1)
        a2 = self.relu_activation(z2)
        z3 = self.eight_to_four(a2)
        a3 = self.relu_activation(z3)
        z4 = self.four_to_two(a3)
        a4 = self.relu_activation(z4)
        z5 = self.two_to_output(a4)
        y_pred = self.relu_activation(z5)

        return y_pred

# Read data and convert to usable data frame format
RedWine_df = pd.read_csv("winequality-red.csv")
formatted_RedWine_dic = {}
with open("winequality-red.csv", mode="r") as file:
    RedWine_csv = csv.reader(file)
    keylist = []
    i = 0
    for rows in RedWine_csv:
        if i == 0:
            rows = str(rows)
            rows = rows.replace("[","")
            rows = rows.replace('"',"")
            rows = rows.replace("'", "")
            rows = rows.replace("]", "")
            rows = rows.split(";")
            for item in rows:
                keylist.append(item)
                formatted_RedWine_dic[item] = []
            i += 1
        else:
            rows = str(rows)
            rows = rows.replace("[", "")
            rows = rows.replace('"', "")
            rows = rows.replace("'", "")
            rows = rows.replace("]", "")
            rows = rows.split(";")
            dict_key = 0
            for item in rows:
                formatted_RedWine_dic[keylist[dict_key]].append(item)
                dict_key += 1
            i += 1
formatted_RedWine_df = pd.DataFrame(formatted_RedWine_dic)

# separate labels from original features
X = formatted_RedWine_df.drop(columns=['quality'])  # Features
y = formatted_RedWine_df['quality']
X = X.astype(float)
y = y.astype(int)

# set size of train and test sizes and split data
test_size = 0.5
num_test_samples = int(test_size * len(X))

# record random split for reproductivity by shuffling indexes
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X))
X_shuffled = X.iloc[shuffled_indices]
y_shuffled = y[shuffled_indices]

X_train = X_shuffled[:-num_test_samples]
X_test = X_shuffled[-num_test_samples:]
y_train = y_shuffled[:-num_test_samples]
y_test = y_shuffled[-num_test_samples:]

# normalization before iteration
mean_values = X_train.mean()
std_dev_values = X_train.std()
X_train_standardized = (X_train - mean_values) / std_dev_values
X_test_standardized = (X_test - mean_values) / std_dev_values

# Defining configurations for training iterations
epochs = 1000
learning_rate = 0.001
model = NeuralNet_30_Neurons()

# setting up for result representation
loss_train = []
loss_test = []

# training and predicting
for epoch in range(epochs):
    Combined_mean_Error = 0.00
    for i in range(len(X_train_standardized)):
        # forward pass prediction on train
        prediction = model.backward(X_train_standardized.iloc[i], np.array(y_train.iloc[i]), learning_rate)

        # calculate loss/error
        error = (prediction - y_train.iloc[i]) ** 2
        Combined_mean_Error += error

    loss = model.RMSE(Combined_mean_Error, y_train)

    Combined_mean_Error = 0.00
    for j in range(len(X_test_standardized)):
        # forward pass prediction on test

        prediction = model.forward(X_test_standardized.iloc[j])
        error = (prediction - y_test.iloc[j]) ** 2
        Combined_mean_Error += error
    loss_t = model.RMSE(Combined_mean_Error, y_test)

    # collecting results for illustration
    loss_test += [loss_t]
    loss_train += [loss]

    # print loss for monitoring training progress
    print(f"Epoch {epoch + 1}/{epochs}", f"Train Error: {loss}", f"Test Error:{loss_t}")

plt.plot(loss_train, label="Train")
plt.title("Train Error Over Epochs - LR = 0.001")
# plt.plot(loss_test, label="Test")
plt.ylabel("Root Mean Squared Error")
plt.xlabel("Iterations")
plt.legend()
plt.show()


