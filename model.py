import pandas as pd
import numpy as np
import sys
import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Model:

    def __init__(self):
        
        print("\n".join(sys.argv))
        
    def preprocessing(self):
        
        """
        LOAD DATA
        """
        infile = sys.argv[1]
        self.df = pd.read_csv(infile, delimiter=',')

        """
        PREPROCESSING DATA
        """
        season=pd.get_dummies(self.df['season'],prefix='season')
        self.df=pd.concat([self.df,season],axis=1)
        weather=pd.get_dummies(self.df['weather'],prefix='weather')
        self.df=pd.concat([self.df,weather],axis=1)
        self.df.drop(['season','weather'],inplace=True,axis=1)
        
        """
        EXTRACT DATE TIME    
        """
        self.df["hour"] = [t.hour for t in pd.DatetimeIndex(self.df.datetime)]
        self.df["dow"] = [t.dayofweek for t in pd.DatetimeIndex(self.df.datetime)]
        self.df['week'] = [t.week for t in pd.DatetimeIndex(self.df.datetime)]
        self.df["month"] = [t.month for t in pd.DatetimeIndex(self.df.datetime)]
        self.df['year'] = [t.year for t in pd.DatetimeIndex(self.df.datetime)]
        self.df['day'] = [t.day for t in pd.DatetimeIndex(self.df.datetime)]
        
        print('Success pre processing data testing\n')
   
    def predict(self):
        
        """
        LOAD MODEL
        """
        loaded_model = pickle.load(open(sys.argv[2], "rb"))
        
        """
        REGRESSION PROCESS
        """
        y_pred = np.exp(loaded_model.predict(self.df.drop(['datetime'],axis=1)))
        
        
        """
        SAVE RESULT
        """
        
        self.df['count'] = y_pred
        self.df.to_csv('answer_model_predict.csv',index=False)



if __name__ == '__main__':
    model = Model()
        
    # Preprocessing Data Testing
    model.preprocessing()
    
    # Predict Data Testing
    model.predict()
