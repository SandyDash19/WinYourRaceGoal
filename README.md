This github repository provides code used in the journal article 
https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-2401-6234.pdf
Citation : S. Dash, "Win Your Race Goal: A Generalized Approach to Prediction of Running Performance,
" Sports Med. Int. Open, vol. 8, p. a24016234, Oct. 2024, doi: 10.1055/a-2401-6234.

This repo contains the following:
1) Dataset mentioned in the article. 
   The csv files can be used for both regression and time series regression analysis.
2) Trainable LSTM model for regression 
3) Trained Hyperparameters

Intructions to run the repo:
1) Clone the repo
2) Change the "dir_path" variable with your local path for the Dataset folder.
3) Change the path in "HPread" variable with your local where HyperParamsLSTMRegression.csv is saved.
4) When you execute the code in a python environment (I used version 3.9.16), you should see print statements for each csv in the Dataset folder.

Output
Results are saved in LSTM_regression.csv which is saved in the folder where you are executing this code. 

If there are any questions please email me at dashs@pdx.edu


