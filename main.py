import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#spremanje baze u df
df = pd.read_csv('heart_attack_prediction_dataset.csv')

df.columns = df.columns.str.strip()

#Problem ne prepoznaje kolonu Patiend ID
#df = df.drop(columns=['Patient ID'])

#shape baze
print("Uzoraka i obelezja ima: ", df.shape)
print('\n\n')

#kolone
print("Kolone: ", df.columns)


print("tipovi vrednosti===========")
numeric_columns = df.select_dtypes(include=np.number)

# Ukupan broj kolona
total_numeric_columns = numeric_columns.shape[1]
print('\n Broj')
print(total_numeric_columns)

print("Kategorije")
cat_columns = df.select_dtypes(include=['object', 'category'])

#Glavni parametri baze
print(df.describe()).T

print('Najmanje:')
print(cat_columns.nunique().idxmin())
print('')
print('Najvise:')
print(cat_columns.nunique().idxmax())

print('')

#8. pitanje
procenat_po_klasi = df['Heart Attack Risk'].value_counts(normalize=True)

print('Procentunalno po Klasi')
print(round(100*procenat_po_klasi))

#Provera da li ima null vrednosti
nulls = df.isnull().sum().sort_values(ascending=False)
print('Nulls: ', nulls) 
#database ima 0 null vrednosti

#10. pitanje

nevalidni = df.isna().any()
print(nevalidni)


print("Outliers")
print("\n")

plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.show()

#15. pitanje

correlacija = df.select_dtypes(include=np.number)
corr = correlacija.corr()
sns.heatmap(corr, annot=True)
plt.show()

#16. pitanje 

for feature in df.columns[:-1]:  # Izbacujemo poslednju kolonu koja je izlazna klasa (HeartDisease)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=feature, hue='Heart Attack Risk', kde=True, alpha= 0.5)
    plt.title(f'Histogram za {feature}')
    plt.show()
