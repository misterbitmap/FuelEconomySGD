import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

fuel = pd.read_csv('dataset/fuel.csv')

X = fuel.copy()
# Remove target
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) 

# About line above:
# This line applies a logarithmic transformation to the target variable y.
# Log transformation is often used for target variables that have a skewed distribution or
# when the relationship between features and target is expected to be multiplicative rather than additive.
# It can help in making the target variable's distribution more normal and can sometimes improve
# model performance.

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))



model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])


#Add Loss and Optimizer

#Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for
#most problems without any parameter tuning (it is "self tuning", in a sense).
#Adam is a great general-purpose optimizer.

model.compile(
    optimizer='adam',
    loss='mae'
)

#Train Model

history = model.fit(
    X,y,
    batch_size=128,
    epochs=200
)

# Get a plot of the training loss.
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5.
history_df.loc[5:, ['loss']].plot();

plt.show()

