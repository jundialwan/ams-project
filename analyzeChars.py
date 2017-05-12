import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv')
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
math = np.mean(train_qs.apply(lambda x: '[math]' in x))
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

print('Pertanyaan dengan tanda tanya (?): {:.2f}%'.format(qmarks * 100))
print('Pertanyaan dengan tag [math]: {:.2f}%'.format(math * 100))
print('Pertanyaan dengan tanda titik (.): {:.2f}%'.format(fullstop * 100))
print('Pertanyaan dengan huruf capital: {:.2f}%'.format(capitals * 100))
print('Pertanyaan dengan huruf pertama yang kapital: {:.2f}%'.format(capital_first * 100))
print('Pertanyaan dengan angka: {:.2f}%'.format(numbers * 100))