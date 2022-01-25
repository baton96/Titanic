# ⚓ Titanic ⚓

This projects was created for the course of "Machine Learning" on the International Hellenic University in Thessaloniki
during my Erasmus stay and consists of a prediction model in Kaggle's *"legendary Titanic ML competition"*, as it was
called by its creators, where *"the competition is simple - use machine learning to create a model that predicts which
passengers survived the Titanic shipwreck*".

More information about this competition can be found on it's [official site](https://www.kaggle.com/c/titanic).

**Obtained accuracy is 82.54 %**

## Goal

The goal is to predict if a passenger survived the sinking of the Titanic or not. For each entry in the test set, you
must make a prediction in a form of either 0 or 1.

## Metric

Score is measured by the percentage of passengers predicted
correctly ([accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)).

## Submission File Format

Submission has to be a csv file with exactly 418 entries plus a header row and will show an error if it has extra
columns (beyond PassengerId and Survived) or rows.

File should have exactly 2 columns:

* PassengerId (sorted in any order)
* Survived (contains your binary predictions: 1 for survived, 0 for deceased)

```
PassengerId,  Survived
892,          0
893,          1
894,          0
...
```
# Data
The data is split into two groups:
* training set ([train.csv](train.csv))
* test set ([test.csv](test.csv))  

| Variable | Definition | Comment |
| --- | --- | --- |
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class | A proxy for socio-economic status <br /> 1 = Upper, 2 = Middle, 3 = Lower |
| sex | Sex | "male" or "female" |
| Age | Age in years | |
| sibsp | Number of (step)siblings and spouses aboard Titanic | Sibling = (step)brother, (step)sister <br /> Spouse = husband, wife <br /> (mistresses and fiancés were ignored)|
| parch | Number of parents and (step)children aboard Titanic | Parent = mother, father <br /> Child = (step)daughter, (step)son |
| ticket | Ticket number | |
| fare | Passenger fare | |
| cabin | Cabin number | |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

# Rules
1. **One account per participant**  
You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts.
2. **No private sharing outside teams**  
Privately sharing code or data outside of teams is not permitted. It's okay to share code if made available to all participants on the forums.
3. **Team Mergers**  
Team mergers are allowed and can be performed by the team leader. In order to merge, the combined team must have a total submission count less than or equal to the maximum allowed as of the merge date. The maximum allowed is the number of submissions per day multiplied by the number of days the competition has been running.
4. **Team Limits**  
There is no maximum team size.
5. **Submission Limits**  
You may submit a maximum of 10 entries per day.  
You may select up to 5 final submissions for judging.
