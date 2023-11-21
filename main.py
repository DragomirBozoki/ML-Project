import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#spremanje baze u df
df = pd.read_csv('lungcancer.csv')

#shape baze
print(df.shape)
print('\n\n')

#Glavni parametri baze
print(df.describe().T)

#Provera da li ima null vrednosti
nulls = df.isnull().sum().sort_values(ascending=False)
print('Nulls: ', nulls) 
#database ima 0 null vrednosti

#Odbacivanje kolona sa nekorisnim podacima
df.drop(['index', 'Patient Id'], inplace=True, axis = 1)

#Plotovanje izlazne Level da bi se videlo koliko vrednosti pripada kojoj grupi
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].pie(df['Level'].value_counts(), labels=df['Level'].value_counts().index)
axs[0].set_title('Distribucija nivoa')

axs[1].pie(df['Gender'].value_counts(), labels=df['Gender'].value_counts().index)
axs[1].set_title('Distribucij a polova')

plt.tight_layout()
plt.show()




#Transformisanje objekt vrednosti u redne za izlaznu y - level
level_mapping = {'High' : 3,
                 'Medium': 2,
                 'Low' : 1}

df['Level'] = df['Level'].map(level_mapping)
print("")

#Gledanje korelisanih vrednosti, da bi znali koje kolone su korisne a koje ne
corr = df.corr()
f = plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True)
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

df.plot.scatter(x='Coughing of Blood', y='Level', ax=axes[0, 0], title='Coughing of Blood vs Risk Level', c= 'black')
df.plot.scatter(x='Genetic Risk', y='Level', ax=axes[0, 1], title='Genetic Risk vs Risk Level', c= 'black')
df.plot.scatter(x='Smoking', y='Level', ax=axes[0, 2], title='Smoking vs Risk Level', c= 'black')
#==============================
df.plot.scatter(x='Fatigue', y='Level', ax=axes[1, 0], title='Fatigue vs Risk Level', c= 'black')
df.plot.scatter(x='Dust Allergy', y='Level', ax=axes[1, 1], title='Dust Allergy vs Risk Level', c= 'black')
df.plot.scatter(x='Chest Pain', y='Level', ax=axes[1, 2], title='Chest Pain vs Risk Level', c= 'black')

plt.tight_layout()
plt.show()

#Izbacivanje izlazne iz skupa X za obuku i trening
X = df.drop(['Level'], axis = 1)

#Definisanje izlazne promenjive, y - nivo sanse da osoba ima kancer pluća
y = df['Level']

#Deljenje skupova X i y u skupove za trening i test,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=1, shuffle=True)

#vrstu modela koji primenjujemo na problem
lr = LogisticRegression(fit_intercept=True)
