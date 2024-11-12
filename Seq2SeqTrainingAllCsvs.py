import torch
import pandas as pd
from LSTM import Seq2SeqLSTM
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from EarlyStop import EarlyStopping
import optuna
import os
import ast
import timeit

"""
The LSTM model described in this code is published in the following paper
S. Dash, "Win Your Race Goal: A Generalized Approach to Prediction of Running Performance,
" Sports Med. Int. Open, vol. 8, p. a24016234, Oct. 2024, doi: 10.1055/a-2401-6234.

This github repo contains the following:
1) Dataset mentioned in the article. 
   The csv files can be used for both regression and time series regression analysis.
2) Trainable LSTM model for regression 
3) Trained Hyperparameters
"""

# Save the dataset in a folder and provide the path
dir_path = "/home/sandy/PacePredictor/JournalPaperWork/WinYourRaceGoal/Dataset"

# Global variables
# Number of inputs
input_size = 3 # Regression = 3, TSR = 4
# Number of outputs
output_size = 1
# Number of hidden layers
num_layers = 1

# This class converts the input numpy arrays to sequences for LSTM
class SequenceDataset(Dataset):
    def __init__(self, input_data, output_data, sequence_length):
        self.input_data = input_data
        self.output_data = output_data
        self.sequence_length = sequence_length
        self.sequences = self.create_inout_sequences()

    def create_inout_sequences(self):
        inout_seq = []
        size = self.output_data.shape[0]
        L = len(self.input_data)
        for i in range(L - self.sequence_length):
            train_seq = self.input_data[i:i+self.sequence_length]
            train_label = self.output_data[i:i+self.sequence_length]
            inout_seq.append((train_seq, train_label))

            #print (f' train_seq {train_seq}\n')
            #print (f' train_label {train_label}\n')
        return inout_seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

"""
Function to normalize inputs to LSTM network between 0 to 1.
Normalization is to prevent parameter explosion
"""
def normalize (input, max, min):  
    input_norm = (input - min) / (max - min)
    return input_norm
"""
Denormalize the outputs from the network so that prediction is in a human
readable format and can be interpreted when compared against the labels. 
"""
def denormalize (input, input_max, input_min):    
    input_denorm = input * (input_max - input_min) + input_min
    return input_denorm

if torch.cuda.is_available():  
    device = torch.device("cuda")
    print('Training on GPU...')
else:
    device = torch.device("cpu")
    print('Training on CPU...')

"""
In main function, the running logs are loaded and split into train, val and test as specified in figure 4
in the journal article. 
All hyperaparameters are trained using Optuna and saved in a csv. (This code is excluded from this repo).
k-fold cross validation was used during hyperparameter search. (This code is also excluded from this repo).
In the main function that csv is read and hyperparams are extracted. 
seq_length - is treated as a hyperparameter.
val_seqlen - is fixed to 2 because there are only 15 runs or races in val and test sets as shown in figure 4
             in the journal article. 
batch_size - hyperparameter which is only used for training. For validation and test sets, entire set is
             treated as one batch
"""    
def PrepareInputs (train_data, val_data, test_data, seq_length, val_seqlen, batch_size):   

    # features for time series regression
    #features = ['Distance_km', 'Elevation_Gain_m', 'Days_Since_Last_Activity', 'Time_min', 'Age']

    #features for regression
    features = ['Distance_km', 'Elevation_Gain_m', 'Time_min', 'Age']

    # Combine train, val, and test data
    all_data = pd.concat([train_data[features], val_data[features], test_data[features]])

    # Calculate global max and min for each feature
    global_max = all_data.max()
    global_min = all_data.min()

    # Normalize each dataset
    train_normalized = train_data[features].apply(lambda x: normalize(x, global_max[x.name], global_min[x.name]))
    val_normalized = val_data[features].apply(lambda x: normalize(x, global_max[x.name], global_min[x.name]))
    test_normalized = test_data[features].apply(lambda x: normalize(x, global_max[x.name], global_min[x.name]))   
    
    x_train = train_normalized.drop ('Time_min', axis = 1)
    y_train = train_normalized['Time_min']    

    x_val = val_normalized.drop('Time_min', axis=1)
    y_val = val_normalized['Time_min']

    x_test = test_normalized.drop('Time_min', axis=1)
    y_test = test_normalized['Time_min']

    # Convert the variables to numpy
    x_train_np = x_train.to_numpy()
    y_train_np = y_train.to_numpy()
    y_train_np = y_train_np.reshape(-1,1)

    x_val_np = x_val.to_numpy()
    y_val_np = y_val.to_numpy()
    y_val_np = y_val_np.reshape(-1,1)

    x_test_np = x_test.to_numpy()
    y_test_np = y_test.to_numpy()
    y_test_np = y_test_np.reshape(-1,1)
    
    dataset = SequenceDataset(x_train_np, y_train_np, seq_length)
    val_dataset = SequenceDataset(x_val_np, y_val_np, val_seqlen)
    test_dataset = SequenceDataset(x_test_np, y_test_np, 2)

    print ("Length of train_dataset, val and test followed by their shapes")
    print (len(dataset.sequences), dataset.output_data.shape)
    print (len(val_dataset.sequences), val_dataset.output_data.shape)
    print (len(test_dataset.sequences), test_dataset.output_data.shape)
    print (f'seq_len {seq_length}')

    # Create a DataLoader
    # for training use trainable hyerparameter batch_size. Training in batches
    # help with improved computational efficiency and stable convergence.
    # for validation and test sets, the entire set is procssed as one batch  
    # Training, validation and test sets are not shuffled because they are 
    # chronologically sequenced as shown in figure 4 in the journal article.   
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset.sequences), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset.sequences), shuffle=False)

    print (len(dataloader), len(val_dataloader), len(test_dataloader))
    
    print (f"""Training size {x_train_np.shape[0]}, 
            Val size {x_val_np.shape[0]}, 
            Test size {x_test_np.shape[0]}""")
    
    return (dataloader, val_dataloader, test_dataloader, global_max, global_min,
            x_train.shape[0], x_val.shape[0],x_test.shape[0])          

"""
This function takes the dataloaders and performs training and evaluation using LSTM which
is described in detail in the journal article.
"""
def train (dataloader, val_dataloader, test_dataloader, global_max, global_min, Activity_Date,
           hidden_size, epochs, patience, lr):

    # Set this to True if you want to plot training vs. validation loss
    plot = True

    #Extract MovingTimeMax and MovingTimeMin from global_max and global_min
    MovingTime_max = global_max.Time_min.max()
    MovingTime_min = global_min.Time_min.min()

    # Instantiate the model, define loss function and optimizer
    # The choice of loss function and optimizer is justified in the journal article.
    model = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers)
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Regularization technique to prevent overtraining during parameter training
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    # Train the model
    TL = []
    VL = []
    val_L1_err = []
       
    for i in range(epochs):
        train_loss = 0.0
        val_loss = 0.0                
        denorm_pred = 0.0
        denorm_Y = 0.0

        stopped_epoch = i

        model.train()
        for seq, labels in dataloader: 

            #print (seq.shape, labels.shape)
            seq = seq.float().to(device)
            labels = labels.float().to(device)      
            optimizer.zero_grad()      

            y_pred = model(seq)

            #print (y_pred.shape)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        TL.append (train_loss / len(dataloader))

        #print(f'epoch: {i:3} loss: {train_loss:10.8f}')

        #initialize Mean_absolute_err       
        total_absolute_errors = 0.0       
        
        model.eval()        
        with torch.no_grad():
            for inputs, targets in val_dataloader:

                inputs = inputs.float().to(device)
                targets = targets.float().to(device)   

                outputs = model(inputs)
                #print (inputs.shape, outputs.shape, targets.shape)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                denorm_pred = denormalize (outputs, MovingTime_max, MovingTime_min)
                denorm_Y = denormalize (targets, MovingTime_max, MovingTime_min)            

                # denorm_pred and denorm_y have 3D shapes just like output which is (total_val_sequences or batch_size, seq_len, num_output_features)
                # We first average errors across seq_len dimension
                total_absolute_errors = torch.mean (torch.abs(denorm_pred.cpu() - denorm_Y.cpu()), axis=1)   
                #print (f'total_absolute_errors shape {total_absolute_errors.shape}')           

        # Now average errors across total_val_sequences dimension
        val_MAE = torch.round(torch.mean(total_absolute_errors) * 100) / 100
                
        VL.append(val_loss / len(val_dataloader))
        val_L1_err.append(val_MAE) 

        # Prevent Overtaining by keeping an eye on validation loss changes
        early_stopping(VL[i], model)
        if early_stopping.early_stop:
            if patience == patience:
                print(f'Early stopping at epoch {i}')
                break    
          
    test_mean_absE_acrossSeqLen = 0.0  # Accumulator for total absolute errors
    
    #Test outside the epoch loop
    model.eval()        
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)   
            outputs = model(inputs)

            print (outputs.shape)
            test_pred = denormalize (outputs, MovingTime_max, MovingTime_min)
            test_Y    = denormalize (targets, MovingTime_max, MovingTime_min)            
           
            # We first average errors across seq_len dimension
            test_mean_absE_acrossSeqLen = torch.mean (torch.abs(test_pred.cpu() - test_Y.cpu()), axis=1)
            
            # calculate absolute percentage error
            # Add a small denominator if Y is almost 0            
            test_ape = torch.abs(test_pred.cpu() - test_Y.cpu()) / (torch.abs(test_Y.cpu()) + 1e-8) * 100
            
    # Now average errors across total_test_sequences dimension
    test_MAE = torch.round(torch.mean(total_absolute_errors) * 100) / 100
    test_mape = torch.round(torch.mean(test_ape)*100) / 100

    # Compute the standard deviation of the absolute errors across total_val_sequences dimension
    test_std_dev = torch.round(torch.std(test_mean_absE_acrossSeqLen) * 100) / 100   

    if plot == True:
        # Plot the loss values
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(TL, label='Train Loss')
        plt.plot(VL, label='Validation Loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        # Plot validation accuracy values
        plt.subplot(1,2,2)
        plt.plot(val_L1_err, label='Mean Absolute Error')
        plt.xlabel('epoch')
        plt.ylabel('Mean Absolute Error (minutes)')
        plt.legend()
        plt.tight_layout()
        plt.show()        

    print(f'ValMAE = {val_L1_err[-1]}, testMAE = {test_MAE}, stopped_epoch = {stopped_epoch}')

    # Return MAE for validation for the last epoch. Val MAE and Test MAE are very close numbers.
    return test_MAE, test_mape, test_std_dev

def main ():

    startTime = timeit.default_timer()   

    # Create an empty DataFrame with the required columns
    results_df = pd.DataFrame(columns=['file_path', 'patience', 'epochs', 'seq_len', 'hidden_size', 'batch_size', 'lr', 'Test_MAE', 'Test_MAPE', 'Test_StdDev'])
    val_seqlen = 2

    # Read hyperparameters from CSV        
    HPread = pd.read_csv('/home/sandy/PacePredictor/JournalPaperWork/WinYourRaceGoal/HyperParamsLSTMRegression.csv')          

    # Loop through each row of the DataFrame
    for index, row in HPread.iterrows():
        hidden_size = row['hidden_size']
        lr = row['lr']
        epochs = row['epochs']
        batch_size = row['batch_size']
        seq_len = row['seq_len']
        patience = row['patience']    
        expected_filename = row['filename']       
        
        # Must match the filename of the hyperparam file with filename in dir_path            
        for file in os.listdir(dir_path): 
            file_path = os.path.join(dir_path, file)
            if file.endswith('.csv') and file == expected_filename:   
                
                df = pd.read_csv(file_path)
                Activity_Date = df['Activity_Date']
                # Remove entries which has 0 distance and 0 pace
                df = df[~((df['Distance_km'] == 0) & (df['Time_min'] == 0))]

                print (f' Total number of runs {df.shape[0]}')
                # Convert Elevation Gain to m 
                df['Elevation_Gain_m'] = df['Elevation_Gain_ft'] * 0.3048     
                # All rows except the last 15 of df_main which does not have test data
                test_data = df.iloc[-15:]  # Last 15 rows for testing
                df_main = df.iloc[:-15]
                train_data = df_main.iloc[:-15] 
                val_data = df_main.iloc[-15:]  # last 15 rows for validation      
              
                (dataloader, val_dataloader, test_dataloader, global_max, global_min, 
	            train_size, val_size, test_size) = PrepareInputs(train_data, val_data, test_data, seq_len, val_seqlen, batch_size)
                test_MAE, test_mape, test_std_dev = train (dataloader, val_dataloader, test_dataloader, global_max, global_min, Activity_Date,
                                                hidden_size, epochs, patience, lr) 
                testMAE = test_MAE.item()
                testMAPE = test_mape.item()
                testStdDev = test_std_dev.item()
                testMAE = "{:.2f}".format(testMAE)
                testMAPE = "{:.2f}".format(testMAPE)
                testStdDev = "{:.2f}".format(testStdDev)
                print(f"""\t testMAE {test_MAE}, testMAPE {testMAPE}, testStdDev {testStdDev}""")
                    
                # Append results to the DataFrame
                new_row = pd.DataFrame([{'file_path': file, 
                                         'patience' : patience,                                                    
                                         'epochs'   : epochs, 
                                         'seq_len'  : seq_len,
                                         'hidden_size': hidden_size,                                                      
                                         'batch_size' : batch_size,
                                         'lr'         : lr, 
                                         'Test_MAE'   : testMAE,
                                         'Test_MAPE'  : testMAPE, 
                                         'Test_StdDev': testStdDev}])
    
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            results_df.to_csv('LSTM_Regression.csv', index=False)

    endTime = timeit.default_timer()
    #total time 
    print (f'Runtime of the program {(endTime - startTime)/60} minutes')    

if __name__ == '__main__':
    main()
