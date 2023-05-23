# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:19:22 2023

@author: 86189
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from skorch import NeuralNetClassifier
import shap
import pickle
import warnings

#%%
device="cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings('ignore')
# Load and preprocess the stock data
data = pd.read_csv(r"C:\Users\86189\Desktop\Quantitative Finance\dataSet2.csv")
data = data[data['Unnamed: 0'] != 'KRAKEN:USDTUSD']
data = data.groupby('Unnamed: 0')
#%%
## Create Sequences
def create_sequences(features, targets, seq_length):
    xs = np.array([features[i:i+seq_length] for i in range(len(features) - seq_length - 1)])
    ys = np.array([targets[i+seq_length-1] for i in range(len(features) - seq_length - 1)])
    return xs, ys
#%%
def get_x_y_sequences(data,sequence_length):
    features = data.iloc[:, [7,10,11,12,13,14,15,16]].values  
    targets = data.iloc[:, 9].values  
    scaler = MinMaxScaler()
    features[np.isinf(features)] = np.finfo(np.float64).max  
    features[np.isnan(features)] = 0
    features_scaled = scaler.fit_transform(features)
    seq_length = sequence_length
    X, y = create_sequences(features_scaled, targets, seq_length)
    return X, y
#%%
sequence_length = 10
def process_group(group_data):
    return get_x_y_sequences(group_data, sequence_length)
dataset = data.apply(process_group).to_dict()
#%%
def perform_train_test_split(X, y):
    splits = []
    step_size = len(X) // 2
    for i in range(step_size, len(X), (step_size // 6)):
        X_train = X[i - step_size:i, :]
        X_test = X[i:i + (step_size // 6), :]

        y_train = y[i - step_size:i]
        y_test = y[i:i + (step_size // 6)]

        if not len(y_test) < 50:
            splits.append((X_train, X_test, y_train, y_test))
    return splits
#%%
def get_split_dataset():
    split_dataset = {}
    for group_name, group_data in dataset.items():
        X, y = group_data
        splits = perform_train_test_split(X, y)
        split_dataset[group_name] = splits
    return split_dataset
split_dataset = get_split_dataset()
#%%
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = self.input_to_hidden(input) + self.hidden_to_hidden(hidden)
        new_hidden = self.tanh(combined)
        return new_hidden


# Define the attention mechanism
#allow model to weight different parts of the input sequence when making predictions.
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        #use nn.Sequential to group the linear layer and the tanh activation function
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )
        # a learnable parameter vector v of size hidden_size. This vector will be used to calculate the attention energies.
        self.v = nn.Parameter(torch.rand(hidden_size))  #generates a tensor filled with random numbers from a nuiform distribution on the inverval[0,1]
        # it will be included in the list of parameters returned by model.parameters(), and its gradients will be updated during backpropagation.
        # When the loss is calculated, the gradients are backpropagated through the network, including the attention mechanism. As a result, the gradient with respect to v is computed, and the optimizer updates the value of v based on this gradient.

    def forward(self, hidden, encoder_outputs):
        hidden_repeated = hidden.repeat(1, encoder_outputs.size(1), 1)
        concat = torch.cat((hidden_repeated, encoder_outputs), 2)
        attn_energies = torch.matmul(self.attn(concat), self.v)
        attn_weights = torch.softmax(attn_energies, dim=1) #normalized
        context = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)
        return context

# Define the stacked RNN model with attention
class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) # 添加 dropout 层
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # initialize hidden state
        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(device) 

        rnn_outputs = []
        for t in range(seq_len):
            hidden = self.rnn_cell(x[:, t, :], hidden)
            hidden = self.dropout(hidden) # 在每次 RNNCell 后应用 dropout
            rnn_outputs.append(hidden)
        rnn_outputs = torch.stack(rnn_outputs, dim=1)

        context = self.attention(hidden.unsqueeze(1), rnn_outputs)
        out = self.fc(context)
        out = self.dropout(out) # 在最后的全连接层后应用 dropout
        return out.squeeze(1)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=True)

#%%
# Set hyperparameters

torch.manual_seed(69)
input_size = 8
hidden_size = 64
learning_rate = 0.001
num_classes = 3 
#%%
def get_global_results(epochs, num_layers, batch_size, dropout):
    global_results = {}
    net = NeuralNetClassifier(
        AttentionRNN,
        module__dropout = dropout,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__num_layers=num_layers,
        module__output_size=num_classes,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        lr=learning_rate,
        batch_size=batch_size,
        max_epochs=epochs,
        device="cuda" if torch.cuda.is_available() else "cpu", 
        train_split=None, # Use GPU if available
        verbose  = 1)
    for group_name, splits in split_dataset.items():
        inner_result = []
        for i, split in enumerate(splits):
            print(f"Processing group {group_name}, split {i+1}")
            
            # Get the data for the split
            X_train, X_test, y_train, y_test = split
            
            y_test[y_test == -1] = 2
            y_train[y_train == -1] = 2
            
            X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype = torch.long)
            
            # Initialize the model
            net.fit(X_train_tensor, y_train_tensor)
    
                 
            # Select a subset of the test data to compute SHAP values
            X_test_subset = X_test  # Adjust as needed
        
            # Convert to PyTorch tensor
            X_test_tensor_subset = torch.tensor(X_test_subset, dtype=torch.float32)
        
            # Initialize the explainer
            explainer = shap.DeepExplainer(net.module_, X_test_tensor_subset)
        
            # Compute SHAP values
            shap_values = explainer.shap_values(X_test_tensor_subset)
        
            # Calculate the average absolute SHAP values for each feature
            feature_importance_positive = np.mean(np.abs(shap_values[1]), axis=0)
            feature_importance_negative = np.mean(np.abs(shap_values[2]), axis =0)
            feature_importance_stable = np.mean(np.abs(shap_values[0]), axis =0)
            
            feature_importance_positive = np.mean(feature_importance_positive, axis=0)
            feature_importance_negative = np.mean(feature_importance_negative, axis =0)
            feature_importance_stable = np.mean(feature_importance_stable, axis =0)
    
            predictions = net.predict(X_test_tensor)
            predictions1 = net.predict(X_train_tensor)
            # Calculate the accuracy
            f1_macro = f1_score(y_test, predictions, average='macro')
            accuracy = accuracy_score(y_test, predictions)
            f1_macro1 = f1_score(y_train, predictions1, average='macro')
            accuracy1 = accuracy_score(y_train, predictions1)
            
            print(f"Accuracy on test data: {accuracy:.4f}")
            print(f"f1_score on test data: {f1_macro:.4f}")
    
            print(f"Accuracy on train data: {accuracy1:.4f}")
            print(f"f1_score on train data: {f1_macro1:.4f}")
            result = {
                'Symbol': group_name,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'predictions': predictions,
                'train_accuracy': accuracy1,
                'train_f1score': f1_macro1,
                'feature_importance_positive': feature_importance_positive,
                'feature_importance_negative': feature_importance_negative,
                'feature_importance_stable': feature_importance_stable
            }
            inner_result.append(result)
        global_results[group_name] = inner_result
    return global_results
#%%
def get_mean_scores_by_currency(best_results):
    currency_accuracies = {}
    currency_f1_scores = {}

    for currency, results in best_results.items():
        total_accuracy = 0
        total_f1_score = 0
        num_splits = len(results)

        for result in results:
            total_accuracy += result['accuracy']
            total_f1_score += result['f1_macro']

        mean_accuracy = total_accuracy / num_splits
        mean_f1_score = total_f1_score / num_splits

        currency_accuracies[currency] = mean_accuracy
        currency_f1_scores[currency] = mean_f1_score

    return currency_accuracies, currency_f1_scores
#%%
params = [
(30, 2, 32, 0.1),
(30, 2, 64, 0.1),
(30, 3, 16, 0.1),
(30, 3, 32, 0.1),
(30, 3, 64, 0.1),
(30, 4, 16, 0.1),
(30, 4, 32, 0.1),
(30, 4, 64, 0.1)]
for epochs, num_layers,batch_size, dropout in params:
    global_results = get_global_results(epochs, num_layers, batch_size, dropout)
    with open(r"C:/Users/86189/Desktop/Quantitative Finance/新建文件夹 (3)/RNN/{}_{}_{}_{}".format(epochs, num_layers, batch_size, dropout), "wb") as f:
        pickle.dump(global_results, f)
    currency_accuracies, currency_f1_scores = get_mean_scores_by_currency(global_results)
    overall_mean_accuracy = []
    overall_mean_f1_macro = []
    for currency, mean_accuracy in currency_accuracies.items():
        mean_f1_score = currency_f1_scores[currency]
        overall_mean_accuracy.append(mean_accuracy)
        overall_mean_f1_macro.append(mean_f1_score)
    overall_mean_accuracy = np.mean(overall_mean_accuracy)
    overall_mean_f1_macro = np.mean(overall_mean_f1_macro)
    currency_accuracies['overall_mean_accuracy'] = overall_mean_accuracy
    currency_f1_scores['overall_mean_f1'] = overall_mean_f1_macro
    with open(r"C:/Users/86189/Desktop/Quantitative Finance/新建文件夹 (3)/RNN_metrics/accuracy_{}_{}_{}_{}".format(epochs, num_layers, batch_size, dropout), "wb") as f:
        pickle.dump(currency_accuracies, f)
    with open(r"C:/Users/86189/Desktop/Quantitative Finance/新建文件夹 (3)/RNN_metrics/F1_{}_{}_{}_{}".format(epochs, num_layers, batch_size, dropout), "wb") as f:
        pickle.dump(currency_f1_scores, f)



