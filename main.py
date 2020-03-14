import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp

from sklearn.preprocessing import StandardScaler, MaxAbsScaler 
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

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
	#sns.heatmap(confusion_matrix(survived,predictions),annot=True,fmt='0')
	output = pd.DataFrame({'PassengerId': testId, 'Survived': predictions})
	output.to_csv(name+'.csv',index=False)
	'''
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	i = 0
	X = train
	for train2, test2 in cv.split(X, y):
	    probas_ = pipeline.fit(X.loc[train2], y.loc[train2]).predict_proba(X.loc[test2])
	    # Compute ROC curve and area the curve
	    fpr, tpr, thresholds = roc_curve(y[test2], probas_[:, 1])
	    tprs.append(interp(mean_fpr, fpr, tpr))
	    tprs[-1][0] = 0.0
	    roc_auc = auc(fpr, tpr)
	    aucs.append(roc_auc)
	    i += 1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
	         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
	         lw=2, alpha=.8)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	'''

train = pd.read_csv("train.csv").sample(frac=1)    
y = train['Survived']
test = pd.read_csv("test.csv").sample(frac=1)    
testId = test.PassengerId
data = train.append(test, sort=False)

test_data_with_labels = pd.read_csv('titanic.csv')[['survived','ticket','name']]
test_data_with_labels.name = test_data_with_labels.name.apply(lambda name: name.replace('"',''))
test.Name = test.Name.apply(lambda name: name.replace('"',''))
survived = pd.merge(test, test_data_with_labels, left_on=['Ticket','Name'], right_on=['ticket','name'], how='left')['survived'].values

data['Sex'] = data['Sex']=='male'
data['Family_Size'] = data['Parch'] + data['SibSp']
#ticketFreq = data.Ticket.value_counts()
#data['Fare'] = data[['Fare','Ticket']].apply(lambda x: x[0]/ticketFreq[x[1]],axis=1)

data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.')
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data.replace({'Title': mapping}, inplace=True)
for title in list(data['Title'].unique()):
	age = data.groupby('Title')['Age'].mean()[title]
	data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age
data.loc[data['Fare'].isnull(), 'Fare'] = data['Fare'].mean()
data['FareBin'] = pd.qcut(data['Fare'], 5).cat.codes
data['AgeBin'] = pd.qcut(data['Age'], 4).cat.codes

data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
data['Family_Survival'] = 0.5
tmp = 0
for _, group in data.groupby(['Last_Name', 'Fare']):    
	if (len(group) > 1):
		tmp += 1
		data.loc[group.index, 'Family_Survival'] = int(group['Survived'].mean() > 0.5)
print(f'There are {tmp} lastNameFare groups')
tmp = 0
for _, group in data.groupby('Ticket'):
	if (len(group) > 1):
		tmp += 1
		for id, row in group.iterrows():
			if (row['Family_Survival'] <= 0.5):
				data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = int(group.drop(id)['Survived'].mean() > 0.5)
print(f'There are {tmp} ticket groups')

data = data[['Pclass', 'Sex', 'Family_Size', 'Family_Survival', 'FareBin', 'AgeBin']]
#sns.heatmap(data.corr(),annot=True,fmt='%2f')
train = pd.DataFrame()
test = pd.DataFrame()
for col in data.columns:
	train[col] = data[col][:891]
	test[col] = data[col][891:]
testAndSave('SVC', SVC(gamma='auto', probability=True))
testAndSave('MLPClassifier', MLPClassifier(hidden_layer_sizes=(10, 10, 10), solver='lbfgs', alpha=10, random_state=1))
testAndSave('LogisticRegression', LogisticRegression(solver='liblinear', random_state=1))
testAndSave('BernoulliNB', BernoulliNB(binarize=None))
