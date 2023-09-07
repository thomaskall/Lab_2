import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns

df1 = pd.read_csv('DF1',index_col=0, header=0)
print("Data Set")
display(df1)

mean = np.mean(df1, axis=0)
print("Column Means")
print(mean)

dc = df1 - mean
print("Centered matrix")
display(dc)

corr_mat = dc.T.dot(dc) / int(dc.shape[0]-1)
print("Correlation Matrix")
display(corr_mat)

# 5. Calculate the correlation matrix
print("\nCorrelation Matrix with NumPy")
print(df1.corr())

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Pairwise Correlation Heatmap')
plt.show()

# Print or use the correlation matrix as needed
#print(correlation_matrix)