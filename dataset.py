import pandas as pd

class CandyDataset():
    def __init__(self,path):
        self.df = pd.read_csv(path)

    def get_classification_data(self):
        return self.df.values[:, 1:-1],(self.df.values[:, -1] >= 52.5).astype(int)

    def get_regression_data(self):
        return self.df.values[:, 1:-1],\
               self.df.values[:, -1]/100
