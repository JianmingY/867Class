import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

class NeuralNet_30_Neurons():
    def __init__(self, N_layer1 = 12, N_layer2 = 8, N_layer3 = 8, N_layer4 = 12):  # defined total 30 hidden neurons
        super(NeuralNet_30_Neurons,self).__init__()

        self.w1 = np.random.random((11, N_layer1))
        self.w2 = np.random.random((N_layer1, N_layer2))
        self.w3 = np.random.random((N_layer2, N_layer3))
        self.w4 = np.random.random((N_layer3, N_layer4))
        self.w5 = np.random.random((N_layer4, 1))

        self.bias = 1


    # Listing the matrix multipliaction between layers
    def eleven_to_twelve(self, X):
        return np.dot(X,self.w1) + self.bias

    def twelve_to_eight(self, X):
        return np.dot(X,self.w2) + self.bias

    def eight_to_eight(self, X):
        return np.dot(X, self.w3) + self.bias

    def eight_to_twelve(self, X):
        return np.dot(X, self.w4) + self.bias

    def twelve_to_one(self, X):
        return np.dot(X, self.w5) + self.bias

    def regression_sigmoid(self, X):
        '''
        Regression - closest class value
        :param X: latent output from the
        :return: class value
        '''
        return 1 / (1 + np.exp(-X))

    def regression_tanh(self, X):
        '''
        Regression - closest class value
        :param X: latent output from the
        :return: class value
        '''
        return np.tanh(X)

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
        z1 = self.eleven_to_twelve(X)
        a1 = self.relu_activation(z1)
        z2 = self.twelve_to_eight(a1)
        a2 = self.relu_activation(z2)
        z3 = self.eight_to_eight(a2)
        a3 = self.relu_activation(z3)
        z4 = self.eight_to_twelve(a3)
        a4 = self.relu_activation(z4)
        z5 = self.twelve_to_one(a4)
        y_pred = self.regression_sigmoid(z5)


        # Backpropagation: backward propagate each class's error, find gradient
        delta5 = y_pred - y_true
        delta4 = np.dot(self.w5, delta5) * (a4 > 0)
        delta3 = np.dot(self.w4, delta4) * (a3 > 0)
        delta2 = np.dot(self.w3, delta3) * (a2 > 0)
        delta1 = np.dot(self.w2, delta2) * (a1 > 0)

        # Update weights and biases
        self.w5 -= learning_rate * np.dot(delta5[:, np.newaxis], a4[np.newaxis, :]).T
        self.w4 -= learning_rate * np.dot(delta4[:, np.newaxis], a3[np.newaxis, :]).T
        self.w3 -= learning_rate * np.dot(delta3[:, np.newaxis], a2[np.newaxis, :]).T
        self.w2 -= learning_rate * np.dot(delta2[:, np.newaxis], a1[np.newaxis, :]).T
        self.w1 -= learning_rate * np.dot(delta1[:, np.newaxis], np.array(X)[np.newaxis, :]).T

        return y_pred

    def forward(self, X):
        z1 = self.eleven_to_twelve(X)
        a1 = self.relu_activation(z1)
        z2 = self.twelve_to_eight(a1)
        a2 = self.relu_activation(z2)
        z3 = self.eight_to_eight(a2)
        a3 = self.relu_activation(z3)
        z4 = self.eight_to_twelve(a3)
        a4 = self.relu_activation(z4)
        z5 = self.twelve_to_one(a4)
        y_pred = self.regression_tanh(z5)

        return y_pred


# Defining 1000 iterations
epochs = 1000
learning_rate = 0.01
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

X = formatted_RedWine_df.drop(columns=['quality'])  # Features
y = formatted_RedWine_df['quality']

X = X.astype(float)
y = y.astype(int)


test_size = 0.5
num_test_samples = int(test_size * len(X))

np.random.seed(42)
shuffled_indices = np.random.permutation(len(X))
X_shuffled = X.iloc[shuffled_indices]

y_shuffled = y[shuffled_indices]

X_train = X_shuffled[:-num_test_samples]
X_test = X_shuffled[-num_test_samples:]
y_train = y_shuffled[:-num_test_samples]
y_test = y_shuffled[-num_test_samples:]

mean_values = X_train.mean()
std_dev_values = X_train.std()


X_train_standardized = (X_train - mean_values) / std_dev_values
X_test_standardized = (X_test - mean_values) / std_dev_values


mean_values = y_train.mean()
std_dev_values = y_train.std()


y_train = (y_train - mean_values) / std_dev_values
y_test = (y_test - mean_values) / std_dev_values


model = NeuralNet_30_Neurons()

loss_train = []
loss_test = []

for epoch in range(epochs):
    Combined_mean_Error = 0.00
    for i in range(len(X_train_standardized)):
        # Forward pass
        prediction = model.backward(X_train_standardized.iloc[i], np.array(y_train.iloc[i]), learning_rate)

        # Calculate loss

        error = (prediction - y_train.iloc[i]) ** 2
        Combined_mean_Error += error
        loss = model.RMSE(Combined_mean_Error, y_train)



    Combined_mean_Error = 0.00
    for j in range(len(X_test_standardized)):
        # Forward pass

        prediction = model.forward(X_test_standardized.iloc[j])

        error = (prediction - y_test.iloc[j]) ** 2
        Combined_mean_Error += error
        loss_t = model.RMSE(Combined_mean_Error, y_test)

    loss_test += [loss_t]
    loss_train += [loss]


    # Print loss for monitoring training progress
    print(f"Epoch {epoch + 1}/{epochs}",loss,loss_t)

plt.plot(loss_train, label="Train")
plt.plot(loss_test, label="Test")
plt.legend()
plt.show()

