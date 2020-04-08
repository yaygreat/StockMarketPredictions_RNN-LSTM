import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
#plotting
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
#normalizing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

seq_len = 30
lr = 0.01

class RNN:
    def __init__(self, X, Y, header = True, hidden_layer_size = 1):
        np.random.seed(1)

        self.hidden_layer_size = hidden_layer_size
        self.X = X
        self.Y = Y
        input_layer_size = 1
        output_layer_size = 1 #len(self.Y)

        # Formula outputs for LSTM
        self.input_activation = np.empty(shape = (len(self.X[0]), hidden_layer_size))
        self.input_gate       = np.empty(shape = (len(self.X[0]), hidden_layer_size))
        self.forget_gate      = np.empty(shape = (len(self.X[0]), hidden_layer_size))
        self.output_gate      = np.empty(shape = (len(self.X[0]), hidden_layer_size))
        self.internal_state   = np.empty(shape = (len(self.X[0]), hidden_layer_size))
        self.H                = np.empty(shape = (len(self.X[0]), hidden_layer_size)) #self.output
        # Weight matrices for LSTM (analogous to Wxh)
        self.W_activation = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        self.W_input      = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        self.W_forget     = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        self.W_output     = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        # Weight matrices for LSTM (analogous to Whh)
        self.U_activation = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        self.U_input      = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        self.U_forget     = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        self.U_output     = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        # Bias vectors for LSTM
        self.b_activation = 2 * np.random.random((hidden_layer_size, 1)) - 1
        self.b_input      = 2 * np.random.random((hidden_layer_size, 1)) - 1
        self.b_forget     = 2 * np.random.random((hidden_layer_size, 1)) - 1
        self.b_output     = 2 * np.random.random((hidden_layer_size, 1)) - 1
        # Weight matrix and bias vector from h to z (aka predicted y)
        self.Why = 2 * np.random.random((output_layer_size, hidden_layer_size)) - 1
        self.by = np.full((output_layer_size, 1), -1)

    def train(self, data, max_iterations = 10, learning_rate = lr):
        print('W_activation', self.W_activation)
        print('W_input', self.W_input)
        print('W_forget', self.W_forget)
        print('W_output', self.W_output)

        print('U_activation', self.U_activation)
        print('U_input', self.U_input)
        print('U_forget', self.U_forget)
        print('U_output', self.U_output)

        print('b_activation', self.b_activation)
        print('b_input', self.b_input)
        print('b_forget', self.b_forget)
        print('b_output', self.b_output)

        self.data = data[seq_len:]

        out = np.empty((0,1), float)
        j=0
        check = True

        for i in range(0,len(self.data)):
            inputs = np.empty((0,seq_len), float)
            inputs = np.append(inputs, self.X[i])
            inputs = np.reshape(inputs, (-seq_len, 1))

            y = self.forward_pass(inputs)
            dy = np.empty((0,1), float)
            delta_y = (self.Y[i]-y)
            y = np.reshape(y, (-1, 1))
            dy = np.append(dy, np.array(delta_y), axis=0)
            j += 1

            self.backward_pass(y, dy, learning_rate)

            out = np.append(out, np.array(y), axis=0)

        print()
        print('W_activation', self.W_activation)
        print('W_input', self.W_input)
        print('W_forget', self.W_forget)
        print('W_output', self.W_output)

        print('U_activation', self.U_activation)
        print('U_input', self.U_input)
        print('U_forget', self.U_forget)
        print('U_output', self.U_output)

        print('b_activation', self.b_activation)
        print('b_input', self.b_input)
        print('b_forget', self.b_forget)
        print('b_output', self.b_output)

        return out

    def tr(self, data, max_iterations = 10, learning_rate = 0.2):
        self.data = data[seq_len:]

        out = np.empty((0,1), float)

        for i in range(0,len(self.data)):
            inputs = np.empty((0,seq_len), float)
            inputs = np.append(inputs, self.X[i])
            inputs = np.reshape(inputs, (-seq_len, 1))
            y = self.forward_pass(inputs)
            dy = np.empty((0,1), float)
            delta_y = (self.Y[i]-y)
            y = np.reshape(y, (-1, 1))

            out = np.append(out, np.array(y), axis=0)

        return out

    def test(self, data, X, Y, max_iterations = 1000, learning_rate = 0.8):

        self.data = data[seq_len:]
        self.X = X
        self.Y = Y

        out = np.empty((0,1), float)
        error=0

        for i in range(0,len(self.data)):
            inputs = np.empty((0,seq_len), float)
            inputs = np.append(inputs, self.X[i])
            inputs = np.reshape(inputs, (-seq_len, 1))

            y = self.forward_pass(inputs)
            delta_y = (self.Y[i]-y)
            y = np.reshape(y, (-1, 1))
            out = np.append(out, np.array(y), axis=0)
            dy = np.empty((0,1), float)
            delta_y = (self.Y[i]-y)
            y = np.reshape(y, (-1, 1))

        return out

    def forward_pass(self, inputs):
        self.last_inputs = inputs

        for t in range(0,len(inputs)):
            x = inputs[t]

            if t == 0:
                out_minus_1 = np.zeros(shape = (len(self.H[t]), 1))
                internal_state_minus_1 = 0


                self.input_activation[t] = tanh(self.W_activation.T * x + np.dot(self.U_activation, out_minus_1) + self.b_activation).reshape(self.hidden_layer_size)

                self.input_gate[t] = sigmoid(self.W_input.T * x  + np.dot(self.U_input, out_minus_1) + self.b_input).reshape(self.hidden_layer_size)

                self.forget_gate[t] = sigmoid(self.W_forget.T * x  + np.dot(self.U_forget, out_minus_1) + self.b_forget).reshape(self.hidden_layer_size)

                self.output_gate[t] = sigmoid(self.W_output.T * x  + np.dot(self.U_output, out_minus_1) + self.b_output).reshape(self.hidden_layer_size)

                self.internal_state[t] = (np.multiply(self.input_activation[t], self.input_gate[t]) + np.multiply(self.forget_gate[t], internal_state_minus_1)).reshape(self.hidden_layer_size)

                self.H[t] = (np.multiply(tanh(self.internal_state[t]), self.output_gate[t])).reshape(self.hidden_layer_size)

            else:
                self.input_activation[t] = tanh(self.W_activation.T * x + np.dot(self.U_activation, self.H[t - 1]).reshape(self.hidden_layer_size, 1) + self.b_activation).reshape(self.hidden_layer_size)

                self.input_gate[t] = sigmoid(self.W_input.T * x + np.dot(self.U_input, self.H[t - 1]).reshape(self.hidden_layer_size, 1) + self.b_input).reshape(self.hidden_layer_size)

                self.forget_gate[t] = sigmoid(self.W_forget.T * x + np.dot(self.U_forget, self.H[t - 1]).reshape(self.hidden_layer_size, 1) + self.b_forget).reshape(self.hidden_layer_size)

                self.output_gate[t] = sigmoid(self.W_output.T * x + np.dot(self.U_output, self.H[t - 1]).reshape(self.hidden_layer_size, 1) + self.b_output).reshape(self.hidden_layer_size)

                self.internal_state[t] = (np.multiply(self.input_activation[t], self.input_gate[t]) + np.multiply(self.forget_gate[t], self.internal_state[t - 1])).reshape(self.hidden_layer_size)

                self.H[t] = (np.multiply(tanh(self.internal_state[t]), self.output_gate[t])).reshape(self.hidden_layer_size)

            y = np.tanh(np.dot(self.Why, self.H[t]) + self.by)
        return y

    def backward_pass(self, out, d_y, learn_rate):
        T = len(self.last_inputs)-1

        d_Why = np.dot(d_y, self.H[T].T)
        d_by = d_y

        delta_h = np.zeros_like(self.H[0])
        d_internal_state = np.zeros_like(self.H[0])
        
        d_a = d_i = d_f = d_o = np.zeros_like(self.H[0])
        d_Ua = d_Ui = d_Uf = d_Uo = np.zeros_like(self.H[0])
        d_Wa = d_Wi = d_Wf = d_Wo = np.zeros_like(self.H[0])
        d_ba = d_bi = d_bf = d_bo = np.zeros_like(self.H[0])

        # Backpropagate through time.
        for t in reversed(range(T)):
            d_h = d_y + delta_h

            if t == T:
                d_internal_state = d_h * self.output_gate[t] * (1 - tanh(self.internal_state[t])**2)
            else:
                d_Ua = d_Ua + d_a * self.H[t]
                d_Ui = d_Ui + d_i * self.H[t]
                d_Uf = d_Uf + d_f * self.H[t]
                d_Uo = d_Uo + d_o * self.H[t]

                d_internal_state = d_h * self.output_gate[t] * (1 - tanh(self.internal_state[t])**2) + d_internal_state * self.forget_gate[t+1]

            d_a = d_internal_state * self.input_gate[t] * (1 -(self.input_activation[t])**2)

            d_i = d_internal_state * self.input_activation[t] * self.input_gate[t] * (1 - self.input_gate[t])

            d_f = d_internal_state * self.internal_state[t-1] * self.forget_gate[t] * (1 - self.forget_gate[t])

            d_o = d_h * tanh(self.internal_state[t]) * self.output_gate[t] * (1-self.output_gate[t])
            
            last_input = np.empty((0,1), float)
            last_input = np.append(last_input, self.last_inputs[t])
            last_input = np.reshape(last_input, (-1, 1))

            d_Wa = d_Wa + d_a * last_input
            d_Wi = d_Wi + d_i * last_input
            d_Wf = d_Wf + d_f * last_input
            d_Wo = d_Wo + d_o * last_input

            d_ba = d_ba + d_a
            d_bi = d_bi + d_i
            d_bf = d_bf + d_f
            d_bo = d_bo + d_o

            delta_ha = self.U_activation * d_a
            delta_hi = self.U_input * d_i
            delta_hf = self.U_forget * d_f
            delta_ho = self.U_output * d_o

            delta_h = delta_ha + delta_hi + delta_hf + delta_ho

        # Clip to prevent exploding gradients.
        for d in [d_Why, d_by, d_a, d_i, d_f, d_o, d_Ua, d_Ui, d_Uf, d_Uo, d_Wa, d_Wi, d_Wf, d_Wo, d_ba, d_bi, d_bf, d_bo]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using gradient descent.
        self.Why = self.Why - learn_rate * d_Why
        self.by = self.by + learn_rate * d_by

        self.W_activation = self.W_activation - learn_rate * d_Wa
        self.W_input = self.W_input - learn_rate * d_Wi
        self.W_forget = self.W_forget - learn_rate * d_Wf
        self.W_output = self.W_output - learn_rate * d_Wo

        self.U_activation = self.U_activation - learn_rate * d_Ua
        self.U_input = self.U_input - learn_rate * d_Ui
        self.U_forget = self.U_forget - learn_rate * d_Uf
        self.U_output = self.U_output - learn_rate * d_Uo

        self.b_activation = self.b_activation + learn_rate * d_ba
        self.b_input = self.b_input + learn_rate * d_bi
        self.b_forget = self.b_forget + learn_rate * d_bf
        self.b_output = self.b_output + learn_rate * d_bo

# sigmoid activation function
def sigmoid(x):
    x = np.array(x, dtype = np.float32)

    return 1 / (1 + np.exp(-x))

# tanh activation function
def tanh(x):
    x = np.array(x, dtype=np.float32)

    return np.tanh(x)

def fill_mean(arr):
    for i in range(0,len(arr)):
        if arr[i][0] == 0:
            arr[i][0] = (arr[i-1][0]+arr[i+1][0])/2
    return arr

def preprocess(X):
    # replaces '.' with '0'
    fill_NaN = SimpleImputer(missing_values='.', strategy='constant', fill_value='0')
    imputed_X = pd.DataFrame(fill_NaN.fit_transform(X))
    imputed_X.columns = X.columns
    imputed_X.index = X.index

    # change {DATE, NASDAQCOM} datatypes to {datetime, float}
    # and make DATE the index
    imputed_X = imputed_X.astype({'NASDAQCOM': 'float64'})
    imputed_X['DATE']=pd.to_datetime(imputed_X.DATE,format='%m/%d/%Y')
    imputed_X.index = imputed_X['DATE']
    imputed_X.drop(['DATE'], inplace=True, axis = 1)
    
    # replaces 0 with mean
    imputed_arr = imputed_X.to_numpy()
    proc_arr = fill_mean(imputed_arr)
    proc_X = pd.DataFrame(proc_arr,columns=['NASDAQCOM'])
    proc_X.columns = imputed_X.columns
    proc_X.index = imputed_X.index

    return proc_X

if __name__ == "__main__":
    df = pd.read_csv('https://garybucket1.s3.amazonaws.com/NASDAQCOM.csv', header=0)
    proc_df = preprocess(df)

    # take only the values of the dataset and standardize them from 0-1
    dataset = proc_df.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(dataset)

    # take first 80% as our training set, next 20% as our validation set
    _80p = round(0.8*len(dataset))
    train = scaled_data[0:_80p, :]
    valid = scaled_data[(_80p):, :]
    test_data = scaled_data[(_80p - seq_len):, :]

    #only take blocks of seq_len days as info that will affect our prediction
    x_train, y_train = [], []
    for i in range(seq_len,len(train)):
        x_train.append(scaled_data[i-seq_len:i,0]) # seq_len days / arr entry
        y_train.append(scaled_data[i,0]) # 61st day prediction
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #TRAINING
    rnn = RNN(x_train, y_train)

    a = rnn.train(train)
    tr_pr = scaler.inverse_transform(a)
    first_sixty = np.zeros((seq_len, 1))
    training_price = np.empty((0,1), float)
    training_price = np.append(training_price, first_sixty)
    training_price = np.append(training_price, tr_pr)
    training_price = np.reshape(training_price, (-1, 1))

    #TESTING
    #only take blocks of seq_len days as info that will affect our prediction
    inputs = scaled_data[len(dataset) - len(valid) - seq_len:]
    inputs = inputs.reshape(-1,1)
    x_test, y_test = [], []
    for i in range(seq_len,len(test_data)):
        x_test.append(test_data[i-seq_len:i,0]) # seq_len days / arr entry
        y_test.append(test_data[i,0]) # 61st day prediction
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    preds = rnn.test(test_data, x_test, y_test)
    closing_price = scaler.inverse_transform(preds)

    # plot our predictions
    train = proc_df[:_80p]
    train['TrainingPrice'] = training_price

    y_true=y_pred=rmse=acc=0
    for i in range(seq_len, len(train)):
        y_true = train['NASDAQCOM'][i]
        y_pred = train['TrainingPrice'][i]
        acc += 100 - np.absolute(y_true - y_pred)*100/y_true
        rmse += np.sqrt(((y_true-y_pred)**2)/(len(train)-seq_len))
    train_acc = acc / (len(train)-seq_len)
    train_rmse = rmse/(len(train)-seq_len)

    out = proc_df[_80p:]
    out['Predictions'] = closing_price
    print(out)
    y_true=y_pred=rmse=acc=0
    for i in range(len(out)):
        y_true = out['NASDAQCOM'][i]
        y_pred = out['Predictions'][i]
        acc += 100 - np.absolute(y_true - y_pred)*100/y_true
        rmse += np.sqrt(((y_true-y_pred)**2)/len(out))
    test_acc = acc / len(out)
    test_rmse = rmse/len(out)
    
    train_price = train[seq_len:]
    print('len train_price', len(train_price))
    print('len tr_pr', len(tr_pr))
    train_price['TrainPrice'] = tr_pr

    print()
    print('train accuracy = ', train_acc)
    print('train rmse = ', train_rmse)
    print('test accuracy = ', test_acc)
    print('test rmse = ', test_rmse)
    plt.plot(train[['NASDAQCOM']])
    plt.plot(train_price[['TrainPrice']])
    plt.plot(out[['NASDAQCOM', 'Predictions']])
    plt.show()
