'''
def extractTitle(n):
	title=rTitle.search(n).group(0)
	if title in ['Dona', 'Countess.','Lady.','Master.','Miss.','Mlle.','Mme.','Mrs.','Ms.']:
		return True
	else:
		return False
	
def extractTitle2(n):
    return regex.search(n).group(0)

def extractTicket(n):
	ticket=rTicket.sub('', n)
	if ticket in ['FCC','PC','SWPP']:
		return "High"
	elif ticket in ['','A','CA','LINE','SOC','SOP','SOPP','SOTONO','SOTONOQ','WC']:
		return "Low"
	else:
		#print(ticket)
		return "Medium"
	
def myTest(name):
	#print(train_X[[name, "Survived"]].groupby([name])["Survived"].value_counts())
	print(train_X[[name, "Survived"]].groupby([name]).mean())
	#return train_X[[name, "Survived"]].groupby([name]).mean()

                
train_y = train_data.Survived
train_X = train_X.assign(AgeMissing=pd.isnull(train_data.Age))
train_X = train_X.assign(Sex = train_data.Sex=='male')
train_X = train_X.assign(Cabin=train_data.Cabin.fillna('U').map( lambda c : c[0] ))
train_X = train_X.assign(CabinMissing=pd.Series(pd.isnull(train_data.Cabin)))
#train_X = train_X.assign(myTicket=pd.Series(train_data.Ticket.apply(lambda n: n.isnumeric())))
train_X = train_X.assign(HasSib=train_data.SibSp>0)
train_X = train_X.assign(HasParch=train_data.Parch>0)
train_X = train_X.assign(SibSp=train_X.SibSp.astype(str))
train_X = train_X.assign(Parch=train_X.Parch.astype(str))
train_X = train_X.assign(NotAlone=train_data[['SibSp', 'Parch']].apply(lambda n: n[0]>0 or n[1]>0, axis=1))
train_X = train_X.assign(Title=train_data.Name.apply(extractTitle))
train_X = train_X.assign(FamilySize=pd.Series(train_data.Parch+train_data.SibSp))

#train_X = train_X.assign(FareBand = pd.cut(train_X.Fare, 3))
#train_X = train_X.assign(SibSpBand = pd.cut(train_X.SibSp, 5))
#train_X = train_X.assign(ParchBand = pd.cut(train_X.Parch, 3))
#train_X = train_X.assign(AgeBand = pd.cut(train_X.Age, 4))
#train_X = train_X.assign(Fare=pd.Series(train_data.Fare.apply(round)))  
#train_X.Embarked.fillna('S', inplace = True)
#train_X.Age.fillna(train_X.Age.mean(), inplace = True) 

test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_X = test_X.assign(AgeMissing=pd.isnull(test_data.Age))
test_X = test_X.assign(Male=test_data.Sex=='male')
#test_X = test_X.assign(Cabin=test_data.Cabin.fillna('U').map( lambda c : c[0] ))
#test_X = test_X.assign(CabinMissing=pd.Series(pd.isnull(test_data.Cabin)))
#test_X = test_X.assign(myTicket=pd.Series(test_data.Ticket.apply(lambda n: n.isnumeric())))
#test_X = test_X.assign(myCabin=pd.Series(test_data.Cabin.apply(lambda n: 'X' if str(n)=='nan' or str(n)[0]=='T' else str(n)[0])))
test_X = test_X.assign(HasSib=test_data.SibSp>0)
test_X = test_X.assign(HasParch=test_data.Parch>0)
#test_X = test_X.assign(SibSp=test_data.SibSp.astype(str))
#test_X = test_X.assign(Parch=test_data.Parch.astype(str))
test_X = test_X.assign(NotAlone=test_data[['SibSp', 'Parch']].apply(lambda n: n[0]>0 or n[1]>0, axis=1))
#test_X = test_X.assign(FamilySize=pd.Series(test_data.Parch+test_data.SibSp))
test_X = test_X.assign(Title=test_data.Name.apply(extractTitle))
test_X.Age.fillna(test_X.Age.mean(), inplace = True) 
test_X.Fare.fillna(test_X.Fare.mean(), inplace = True) 
test_X = pd.get_dummies(test_X)

svc = SVC(gamma='auto', random_state=1, probability=True)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), solver='lbfgs', alpha=10, random_state=1)
log = LogisticRegression(solver='liblinear', random_state=1)

pipeline = make_pipeline(SimpleImputer(), StandardScaler(), svc)
scores = cross_val_score(pipeline, train_X, train_y, cv=5)
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))

from sklearn.ensemble import VotingClassifier
estimators=[('svc', svc), ('mlp', mlp), ('log', log)]
ensemble = VotingClassifier(estimators, voting='soft', n_jobs=-1)

from imblearn.over_sampling import SVMSMOTE
pipeline = make_pipeline(SimpleImputer(), SVMSMOTE(random_state=1), StandardScaler(), mlp)

#Pclass, Embarked without Q, SibSp2, BigParch, Kid, Old, BigFare, NoFare, Sex2, CabinMissing, AgeMissing, Roommate, ?Title, ?Ticket
features = ['Pclass', 'Embarked','Ticket', 'Name']
#features = ['Survived']
rTitle = re.compile("[a-zA-Z]+\.")
rRoommate = re.compile("\(")
rTicket = re.compile("[/\.0-9 ]*")
rAlias = re.compile('"')

train_data = pd.read_csv('train.csv').sample(frac=1, random_state=1) 
train_y = train_data.Survived
train_X = train_data[features]

train_X = train_X.assign(Title=train_data.Name.apply(extractTitle))
train_X = train_X.assign(Survived=train_data.Survived)
train_X = train_X.assign(FamilyTicket=train_X[['Name','Ticket','Title']].apply(lambda x: x['Ticket'][:-2]+'XX' if x['Title'] else 'no group', axis=1))
FamilyTicketFreq = train_X.FamilyTicket.value_counts()
FamilyTicketMean = train_X[['FamilyTicket', "Survived"]].groupby(['FamilyTicket']).mean()
FamilyTicketMean = dict(zip(FamilyTicketMean.index, FamilyTicketMean.Survived))
train_X = train_X.assign(FamilyTicketDie =train_X.FamilyTicket.apply(lambda x: False if x not in FamilyTicketMean or x=='no group' or (x in FamilyTicketMean and FamilyTicketMean[x]>=0.5) else True))
train_X = train_X.assign(FamilyTicketLive=train_X.FamilyTicket.apply(lambda x: False if x not in FamilyTicketMean or x=='no group' or (x in FamilyTicketMean and FamilyTicketMean[x]<=0.5) else True))
train_X.drop(['FamilyTicket','Name','Ticket','Survived'], axis=1, inplace=True)

train_X = train_X.assign(SibSp2=train_data.SibSp.apply(lambda x: 2 if x<2 else x))
#train_X = train_X.assign(NoSibSp=train_data.SibSp==0)
#train_X = train_X.assign(BigParch=train_data.Parch>4)
train_X = train_X.assign(Kid=train_data.Age<7)
train_X = train_X.assign(Old=train_data.Age>63)
train_X = train_X.assign(BigFare=train_data.Fare>=120)
train_X = train_X.assign(NoFare=train_data.Fare==0)
train_X = train_X.assign(Sex2=train_data.Sex=='male')
train_X = train_X.assign(CabinMissing=pd.isnull(train_data.Cabin))
train_X = train_X.assign(AgeMissing=pd.isnull(train_data.Age))
#train_X = train_X.assign(Roommate=train_data.Name.apply(lambda n: rRoommate.search(n) != None))
#train_X = train_X.assign(Alias=train_data.Name.apply(lambda n: rAlias.search(n) != None))
#train_X = train_X.assign(Title=train_data.Name.apply(extractTitle))
#train_X = train_X.assign(Ticket=train_data.Ticket.apply(lambda n: extractTicket(n)))
train_X = train_X.assign(NotAlone=train_data[['SibSp', 'Parch']].apply(lambda n: n[0]>0 or n[1]>0, axis=1))
#train_X = train_X.assign(HasSib=train_data.SibSp>0)
#train_X = train_X.assign(HasParch=train_data.Parch>0)
train_X = pd.get_dummies(train_X)
train_X.drop('Embarked_Q', axis=1, inplace=True)
#train_X.drop('Title_Medium', axis=1, inplace=True)
#train_X.drop('Ticket_Medium', axis=1, inplace=True)

test_data = pd.read_csv('test.csv').sample(frac=1, random_state=1) 
test_X = test_data[features]

test_X = test_X.assign(Title=test_data.Name.apply(extractTitle))
test_X = test_X.assign(FamilyTicket=test_X[['Name','Ticket','Title']].apply(lambda x: x['Ticket'][:-2]+'XX' if x['Title'] else 'no group', axis=1))
test_X = test_X.assign(FamilyTicketDie =test_X.FamilyTicket.apply(lambda x: False if x not in FamilyTicketMean or x=='no group' or (x in FamilyTicketMean and FamilyTicketMean[x]>=0.5) else True))
test_X = test_X.assign(FamilyTicketLive=test_X.FamilyTicket.apply(lambda x: False if x not in FamilyTicketMean or x=='no group' or (x in FamilyTicketMean and FamilyTicketMean[x]<=0.5) else True))
test_X.drop(['FamilyTicket','Name','Ticket'], axis=1, inplace=True)

test_X = test_X.assign(SibSp2=test_data.SibSp.apply(lambda x: 2 if x<2 else x))
#test_X = test_X.assign(NoSibSp=test_data.SibSp==0)
#test_X = test_X.assign(BigParch=test_data.Parch>4)
test_X = test_X.assign(Kid=test_data.Age<7)
test_X = test_X.assign(Old=test_data.Age>63)
test_X = test_X.assign(BigFare=test_data.Fare>=120)
test_X = test_X.assign(NoFare=test_data.Fare==0)
test_X = test_X.assign(Sex2=test_data.Sex=='male')
test_X = test_X.assign(CabinMissing=pd.isnull(test_data.Cabin))
test_X = test_X.assign(AgeMissing=pd.isnull(test_data.Age))
#test_X = test_X.assign(Roommate=test_data.Name.apply(lambda n: rRoommate.search(n) != None))
#test_X = test_X.assign(Alias=test_data.Name.apply(lambda n: rAlias.search(n) != None))
test_X = test_X.assign(Title=test_data.Name.apply(extractTitle))
#test_X = test_X.assign(Ticket=test_data.Ticket.apply(lambda n: extractTicket(n)))
test_X = test_X.assign(NotAlone=test_data[['SibSp', 'Parch']].apply(lambda n: n[0]>0 or n[1]>0, axis=1))
#test_X = test_X.assign(HasSib=test_data.SibSp>0)
#test_X = test_X.assign(HasParch=test_data.Parch>0)
test_X = pd.get_dummies(test_X)
test_X.drop('Embarked_Q', axis=1, inplace=True)
#test_X.drop('Title_Medium', axis=1, inplace=True)
#test_X.drop('Ticket_Medium', axis=1, inplace=True)

print(list(set(train_X.columns)-set(test_X.columns)))
print(list(set(test_X.columns)-set(train_X.columns)))

#model = SVC(gamma='scale', random_state=1, probability=True)
model = MLPClassifier(hidden_layer_sizes=(10,10,10), solver='lbfgs', alpha=10, random_state=1)
#model = LogisticRegression(solver='liblinear', random_state=1)

pipeline = make_pipeline(SimpleImputer(), StandardScaler(), model)
scores = cross_val_score(pipeline, train_X, train_y, cv=StratifiedKFold(20, shuffle=True, random_state=1))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))

pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Group3.csv',index=False)
'''
""