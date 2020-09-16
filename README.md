# CONTEXT
&nbsp;&nbsp;&nbsp;&nbsp;This projects was created for the course of "Machine Learning" on the International Hellenic University in Thessaloniki during my Erasmus stay and consists of a prediction model in Kaggle's *"legendary Titanic ML competition"*, as it was called by its creators, where *"the competition is simple - use machine learning to create a model that predicts which passengers survived the Titanic shipwreck*".

&nbsp;&nbsp;More information about this competition can be found on it's [official site](https://www.kaggle.com/c/titanic).
## Goal
&nbsp;&nbsp;The goal is to predict if a passenger survived the sinking of the Titanic or not. For each entry in the test set, you must make a prediction in a form of either 0 or 1.
## Metric
&nbsp;&nbsp;Score is measured by the percentage of passengers predicted correctly ([accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)).
## Submission File Format
&nbsp;&nbsp;Submission has to be a csv file with exactly 418 entries plus a header row and will show an error if it has extra columns (beyond PassengerId and Survived) or rows.

&nbsp;&nbsp;File should have exactly 2 columns:
* PassengerId (sorted in any order)
* Survived (contains your binary predictions: 1 for survived, 0 for deceased)
```
PassengerId,Survived
892,0
893,1
894,0
Etc.
```
# DATA
