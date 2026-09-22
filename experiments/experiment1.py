from data_handling.utils import  load_raw_data
import pandas as pd
#load processed dataset 

training_data_set_path = "dataset/processed_dataset/processed_train.csv"
training_data = load_raw_data(training_data_set_path)



#train vae 



# extract encode from vae and train the vae+gradient boosting 


# train gradient boosting model


# evalute the two models and compare the results


# save the results in experiment1_results folder


