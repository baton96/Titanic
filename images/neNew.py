'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
import seaborn as sns
np.random.seed(0)

train = pd.read_csv("../train.csv").sample(frac=1)    
y = train['Survived']
test = pd.read_csv("../test.csv").sample(frac=1)    
testId = test.PassengerId
data = train.append(test, sort=True)
#data['AgeMissing'] = data['Age'].isnull()
#sns.countplot(x='AgeMissing', data=data, hue='Survived')

sns.kdeplot(data=data.loc[data['Survived']==1,'Fare'], label='Survived')
sns.kdeplot(data=data.loc[data['Survived']==0,'Fare'], label='Not survived')
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MaxAbsScaler 
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
np.random.seed(0)

train = pd.read_csv("../train.csv").sample(frac=1)    
y = train['Survived']
test = pd.read_csv("../test.csv").sample(frac=1)    
testId = test.PassengerId
data = train.append(test, sort=False)
data['Sex'] = data['Sex']=='male'
data['Family_Size'] = data['Parch'] + data['SibSp']

data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.')
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data.replace({'Title': mapping}, inplace=True)
data.loc[data['Fare'].isnull(), 'Fare'] = data['Fare'].mean()
for title in list(data['Title'].unique()):
	age = data.groupby('Title')['Age'].mean()[title]
	data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age
data['FareBin'] = pd.qcut(data['Fare'], 5).cat.codes
data['AgeBin'] = pd.qcut(data['Age'], 4).cat.codes
data = data[['Pclass', 'Sex', 'Family_Size', 'FareBin', 'AgeBin']]
#sns.heatmap(data.corr(),annot=True,fmt='%04.2f')