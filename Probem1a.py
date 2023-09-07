import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

df1 = pd.read_csv('DF1',index_col=0, header=0)
print("Data Set")
display(df1)

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Pairwise Correlation Heatmap')
plt.show()