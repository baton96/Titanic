import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


def test_n_save(name, model):
    pipeline = make_pipeline(SimpleImputer(), StandardScaler(), model)
    pipeline.fit(train, y)
    scores = cross_val_score(pipeline, train, y, cv=KFold(20))
    print(name, "Cross-validation accuracy:%f" % scores.mean())

    predictions = pipeline.predict(test)
    acc = sum(s == p for s, p in zip(survived, predictions)) / len(predictions)
    print(name, "Real accuracy: %f" % acc)

    # output = pd.DataFrame({'PassengerId': testId, 'Survived': predictions})
    # output.to_csv(name + '.csv', index=False)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data = pd.concat([train, test])
testId = test.PassengerId
y = train['Survived']

test_data_with_labels = pd.read_csv('titanic.csv')[['survived', 'ticket', 'name']]
test_data_with_labels.name = test_data_with_labels.name.apply(lambda name: name.replace('"', ''))
test.Name = test.Name.apply(lambda name: name.replace('"', ''))
survived = pd.merge(
    test, test_data_with_labels,
    left_on=['Ticket', 'Name'],
    right_on=['ticket', 'name'],
    how='left'
)['survived'].values

data['Family_Size'] = data['Parch'] + data['SibSp']
data['Sex'] = data['Sex'] == 'male'

data['Title'] = data['Name'].str.extract(r'([A-Za-z]+)\.')
mapping = {
    'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
    'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'
}
data.replace({'Title': mapping}, inplace=True)
for title in list(data['Title'].unique()):
    mean_age_per_title = data.groupby('Title')['Age'].mean()[title]
    idxs = (data['Age'].isnull()) & (data['Title'] == title)
    data.loc[idxs, 'Age'] = mean_age_per_title
data.loc[data['Fare'].isnull(), 'Fare'] = data['Fare'].mean()
data['FareBin'] = pd.qcut(data['Fare'], 5).cat.codes
data['AgeBin'] = pd.qcut(data['Age'], 4).cat.codes

data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
data['Family_Survival'] = 0.5

for _, group in data.groupby(['Last_Name', 'Fare']):
    if len(group) <= 1:
        continue
    family_survival = int(group['Survived'].mean() > 0.5)
    data.loc[group.index, 'Family_Survival'] = family_survival

for _, group in data.groupby('Ticket'):
    if len(group) <= 1:
        continue
    for idx, row in group.iterrows():
        if row['Family_Survival'] > 0.5:
            continue
        family_survival = int(group.drop(idx)['Survived'].mean() > 0.5)
        idxs = data['PassengerId'] == row['PassengerId']
        data.loc[idxs, 'Family_Survival'] = family_survival

data = data[['Pclass', 'Sex', 'Family_Size', 'Family_Survival', 'FareBin', 'AgeBin']]
train = pd.DataFrame()
test = pd.DataFrame()
for col in data.columns:
    train[col] = data[col][:891]
    test[col] = data[col][891:]

test_n_save('LabelPropagation', LabelPropagation('knn', n_neighbors=10, n_jobs=-1))
test_n_save('LabelSpreading', LabelSpreading('knn', n_neighbors=10, n_jobs=-1))
