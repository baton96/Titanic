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

def testAndSave(name, model):
	if(name=='BernoulliNB'): scaler =  MaxAbsScaler()
	else: scaler =  StandardScaler()
	pipeline = make_pipeline(SimpleImputer(), scaler, model)
	pipeline.fit(train, y)
	cv = StratifiedKFold(20, shuffle=True)
	scores = cross_val_score(pipeline, train, y,  cv=cv)
	print(name+" Cross-validation accuracy: %f" % scores.mean())
	predictions = pipeline.predict(test)
	print(name+" Real accuracy: %f" % np.equal(survived, predictions).mean())

train = pd.read_csv("../train.csv").sample(frac=1)    
y = train['Survived']
test = pd.read_csv("../test.csv").sample(frac=1)    
testId = test.PassengerId
data = train.append(test, sort=False)

test_data_with_labels = pd.read_csv('../titanic.csv')[['survived','ticket','name']]
test_data_with_labels.name = test_data_with_labels.name.apply(lambda name: name.replace('"',''))
test.Name = test.Name.apply(lambda name: name.replace('"',''))
survived = pd.merge(test, test_data_with_labels, left_on=['Ticket','Name'], right_on=['ticket','name'], how='left')['survived'].values

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
	#fare = data.groupby('Title')['Fare'].mean()[title]
	#data.loc[(data['Fare'].isnull()) & (data['Title'] == title), 'Fare'] = fare
data['FareBin'] = pd.qcut(data['Fare'], 5).cat.codes
data['AgeBin'] = pd.qcut(data['Age'], 4).cat.codes

data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
data['Family_Survival'] = 0.5
for _, group in data.groupby(['Last_Name', 'Fare']):    
	if (len(group) > 1):
		for id, row in group.iterrows():
			data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = int(group.drop(id)['Survived'].mean() > 0.5)
for _, group in data.groupby('Ticket'):
	if (len(group) > 1):
		for id, row in group.iterrows():
			if (row['Family_Survival'] <= 0.5):
				data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = int(group.drop(id)['Survived'].mean() > 0.5)
data = data[['Pclass', 'Sex', 'Family_Size', 'Family_Survival', 'FareBin', 'AgeBin']]

train = pd.DataFrame()
test = pd.DataFrame()
for col in data.columns:
	train[col] = data[col][:891]
	test[col] = data[col][891:]
testAndSave('SVC', SVC(gamma='auto', probability=True))