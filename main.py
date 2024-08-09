import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset = load_diabetes()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

X = df['bmi'].to_numpy()
y = df['s6'].to_numpy()

w = np.random.randn()
b = np.random.randn()

epochs = 1000
learning_rate = 0.001

for i in range(epochs):
    loss = 0

    for j in range(len(X)):
        y_pred = X[j] * w + b

        loss = np.pow(y_pred - y[j], 2)
        
        dL_dW = 2 * (y_pred - y[j]) * X[j]
        dL_dB = 2 * (y_pred - y[j])

        w = w - dL_dW * learning_rate
        b = b - dL_dB * learning_rate

    print(f"Epoch: {i}/{epochs}, loss : {loss}")

print("Training finished, showing the plot...")

sns.scatterplot(x=X, y=y)

y_pred = (X * w + b)

sns.lineplot(x=X, y=y_pred)

plt.show()