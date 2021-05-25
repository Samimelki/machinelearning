#Import pandas and model_selection module of scikit_learn
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    #Training data is in a CSV file called train.csv
    df= pd.read_csv("mnist_train.csv")

    #Create a new kfold column and fill it with -1
    df["kfold"]=-1

    #randomize the rows of data
    df= df.sample(frac=1).reset_index(drop=True)

    #initiate the kfold class from model_selection module
    kf= model_selection.KFold(n_splits=5)

    #fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold
    
    #save the new csv file with kfold column
    df.to_csv("mnist_train_folds.csv", index=False)
