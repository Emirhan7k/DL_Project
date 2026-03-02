import torch 
import torch.nn as nn
import torch.optim as optim 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 3 farklı sınıfa ait sınıflandırma problemi
df = pd.read_csv('iris.data')

X = df.iloc[:, :-1].values
y,_ = pd.factorize(df.iloc[:, -1])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def to_tensor(data,target):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.long)
    return data_tensor, target_tensor

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)

