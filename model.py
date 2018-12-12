import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
#from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

infile = 'data.csv'

class Model:

    def __init__(self):
        """
        You can add more parameter here to initialize your model
        """
        pass
    
    def train(self):
        """
        NOTE: Implement your training procedure in this method.
        """
        
        """
        LOAD DATA
        """
        df = pd.read_csv(infile, delimiter=',')

        """
        PREPROCESSING DATA
        """
        #remove missing value
        df = df.drop(df.index[df[df['article_content'].isnull()].index])
        #remove URL
        df['article_content'] = df['article_content'].str.replace('((www\.[^\s]+)|(https?://[^\s]+))','')
        #remove \n on article
        df.article_content = df.article_content.replace('\n','', regex=True)
        
        #split dataset
        X_train, X_test, y_train, y_test = train_test_split(df['article_content'],
                                                            df['article_topic'], random_state=1)   

        """
        FEATURE EXTRACTION - Bag Of Word - CountVectorizer
        X_train will fit into vocabulary using counvectorizer
        then it will transform into matrix
        """
        self.vectorizer = CountVectorizer().fit(X_train)
        X_train_count = self.vectorizer.transform(X_train)
        
        """
        CLASSIFICATION - Logistic Regression
        using C parameter = 0.001
        using L2 regularization because all of feature are considered
        """
        self.classifier = LogisticRegression(C=0.01)
        self.classifier.fit(X_train_count, y_train)
        pred = self.classifier.predict(self.vectorizer.transform(X_test))
        
        print("Training set score: {:.2f}".format(self.classifier.score(X_train_count, y_train))) 
        print("Test set score: {:.2f}".format(self.classifier.score(self.vectorizer.transform(X_test), y_test)))
        print("Micro average f1 score: {:.3f}".format (f1_score(y_test, pred, average="micro")))

   
    def predict(self,text):
        """
        NOTE: Implement your predict procedure in this method.
        """
        predictions = self.classifier.predict(self.vectorizer.transform([text]))
        return str(predictions[0])

    def save(self):
        """
        Save trained model to model.pickle file.
        """
        ds.model.save(self, "model-C01.pickle")


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model()
        
    # Train your model
    model.train()
    
    # Save your trained model to model.pickle
    model.save()

