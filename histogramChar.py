import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss

pal = sns.color_palette()

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len) #for char count
dist_test = test_qs.apply(len) #for char count

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Histogram jumlah karakter pada pertanyaan', fontsize=14)
plt.legend()
plt.xlabel('Jumlah char', fontsize=12)
plt.ylabel('Probabilitas', fontsize=12)
plt.show()