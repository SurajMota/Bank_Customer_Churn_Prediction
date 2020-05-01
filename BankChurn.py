#Importing Liab
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pickle


dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head(5)


print("All Columns names are as follows:- ",dataset.columns)
print("------------------------------------------------------------------------")
print("Rows and columns are:-",dataset.shape)
print("------------------------------------------------------------------------")
print("dtypes of all columns are as follows -",dataset.dtypes)

#dataset.isnull().sum()

#Check unique in eaeh column
dataset.nunique()

#checking for unique values in Geography
dataset["Geography"].unique()

# Percentage per category for the target column.
percentage_labels = dataset['Exited'].value_counts(normalize = True) * 100
percentage_labels
#Conclusion is that our output data is imbalanced

#Feature Selection
dataset.drop('EstimatedSalary', axis=1, inplace=True)
dataset.drop('HasCrCard', axis=1, inplace=True)
dataset.drop('Tenure', axis=1, inplace=True)

X = dataset.iloc[:,3:10].values
y = dataset.iloc[:,10].values

#Handling categorical values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X= onehotencoder.fit_transform(X).toarray()
X= X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
print("X_train is - ", X_train.shape)
print("X_test is - ", X_test.shape)
print("y_train is - ", y_train.shape)
print("y_test is - ", y_test.shape)


#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Customers distribution across Countries
plt.figure(figsize = (15,15))
sns.catplot(x = 'Geography', kind = 'count', data = dataset, palette = 'pink')
plt.title('Customers distribution across Countries')
plt.show()

#Males vs Females
plt.figure(figsize = (15,15))
sns.catplot(x = 'Gender', kind = 'count', data = dataset, palette = 'pastel')
plt.title("Males vs Females")
plt.show()

#Active VS Non-Active Members
plt.figure(figsize = (15,15))
sns.catplot(x = 'IsActiveMember', kind = 'count', data = dataset, palette = 'pink')
plt.title("Active VS Non-Active Members")
plt.show()

#Gender and Active Members
plt.figure(figsize = (15,15))
sns.catplot(x = 'IsActiveMember', kind = 'count', hue = 'Gender', palette = 'pink', data = dataset)
plt.title("Gender and Active Members")
plt.show()

plt.figure(figsize = (6,6))
sns.catplot(x = 'Exited', kind = 'count', hue = 'Gender', palette = 'pink', data = dataset)
plt.title("Gender and Exited")
plt.show()

plt.figure(figsize = (6,6))
sns.distplot(dataset['Age'])
plt.title("Age")
plt.show()

plt.figure(figsize = (6,6))
sns.distplot(dataset["CreditScore"])
plt.title("Credit Score")
plt.show()

plt.figure(figsize = (6,6))
sns.distplot(dataset["Balance"])
plt.title("Balance")
plt.show()

#Detecting the Outliers through Box Plot
column = ["Age", "Balance", "EstimatedSalary", "CreditScore"]
for i in column:
    plt.figure(figsize = (6,6))
    sns.boxplot(dataset[i])
    plt.title('Box Plot')
    plt.show()

#Check for Co-relation
plt.figure(figsize=(20,8))
sns.heatmap(dataset.corr(), annot=True, fmt='.0%',cmap="YlGnBu")

#Logistic Regression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_clf.score(X_train, y_train)

y_pred_log = log_clf.predict(X_test)
#Evaluating the Random Forest Model
print("accuracy_score for Logistic Regression : ",accuracy_score(y_test,y_pred_log))
print()
print(confusion_matrix(y_test,y_pred_log))
print()
print(classification_report(y_test,y_pred_log))

# ROC Curve

y_pred_prob = log_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test,y_pred_log)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.legend()

roc_auc_score(y_test,y_pred_log)

#Hypermater tuning of Logistic regression
from sklearn.model_selection import GridSearchCV

grid = {'penalty': ['l1', 'l2'],'C':np.logspace(-3,3,7)}

tuning = GridSearchCV(estimator=log_clf,param_grid=grid,scoring='accuracy',cv=10, refit=True, n_jobs=-1)

tuning.fit(X_train, y_train)
print(tuning.best_params_)
print(tuning.best_score_)

#Logistic Regression Model after Hyperparameter tuning through GridSearch
from sklearn.linear_model import LogisticRegression
log_clf_tunn = LogisticRegression(penalty='l1',C=1.0)
log_clf_tunn.fit(X_train, y_train)

log_clf_tunn.score(X_train, y_train)
y_pred_log_tunn = log_clf_tunn.predict(X_test)
#Evaluating the Logistic Regression after Hyperparameter tuning Model
print("accuracy_score for Logistic Regression : ",accuracy_score(y_test,y_pred_log_tunn))
print()
print(confusion_matrix(y_test,y_pred_log_tunn))
print()
print(classification_report(y_test,y_pred_log_tunn))

# ROC Curve

y_pred_prob = log_clf_tunn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test,y_pred_log_tunn)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.legend()

roc_auc_score(y_test, y_pred_log_tunn)





#Decision tree Classification
tree_clf = DecisionTreeClassifier(max_depth=8)
tree_clf.fit(X_train, y_train)
tree_clf.score(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)

#Evaluating the Decision tree Classifier Model
print("accuracy_score for Decision Tree : ",accuracy_score(y_test,y_pred_tree))
print()
print(confusion_matrix(y_test,y_pred_tree))
print()
print(classification_report(y_test,y_pred_tree))




#Random Forest Classifier
rmd_clf = RandomForestClassifier(max_depth=17)
rmd_clf.fit(X_train, y_train)
rmd_clf.score(X_train, y_train)

y_pred_forest = rmd_clf.predict(X_test)

#Evaluating the Random Forest Model
print("accuracy_score for Random Forest : ",accuracy_score(y_test,y_pred_forest))
print()
print(confusion_matrix(y_test,y_pred_forest))
print()
print(classification_report(y_test,y_pred_forest))


#Hyper Parameter Tuning of random forest
#from sklearn.model_selection import RandomizedSearchCV

#n_est = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
#m_depth = [int(x) for x in np.linspace(5, 50, num = 5)]
#min_samp = [3, 5, 6, 7, 10, 11]
#m_ftr = ['auto']


#param_grid = {'max_depth': m_depth, 'max_features': m_ftr,'n_estimators': n_est,'min_samples_split': min_samp}
#RF_cv = RandomizedSearchCV(estimator = RandomForestClassifier(),
                          # n_iter=100,
                          # param_distributions =  param_grid,
                          # random_state=51,
                          # cv=3,
                          # n_jobs=-1,
                          # refit=True)

#RF_cv.fit(X_train,y_train)

# Evaluate model with ROC curve
y_pred_prob = rmd_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_forest)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Random Forest')
plt.legend()

# Evaluate model with ROC curve Percent
roc_auc_score(y_test, y_pred_forest)





#Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(max_depth=8)
gb_clf.fit(X_train, y_train)
gb_clf.score(X_train, y_train)

y_pred_gb = rmd_clf.predict(X_test)

#Evaluating the Gradient Boosting Model
print("accuracy_score for Gradient Boosting Model : ",accuracy_score(y_test,y_pred_gb))
print()
print(confusion_matrix(y_test,y_pred_gb))
print()
print(classification_report(y_test,y_pred_gb))

# Evaluate model with ROC curve
y_pred_prob = gb_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_gb)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Gradient Boosting')
plt.legend()

roc_auc_score(y_test, y_pred_gb)




#Adda Boost Classifier
ada_clf = AdaBoostClassifier(n_estimators=100,learning_rate=1.0)
ada_clf.fit(X_train, y_train)
ada_clf.score(X_train, y_train)

y_pred_ada = ada_clf.predict(X_test)

#Evaluating the Gradient Boosting Model
print("accuracy_score for AdaBoost Model : ",accuracy_score(y_test,y_pred_ada))
print()
print(confusion_matrix(y_test,y_pred_ada))
print()
print(classification_report(y_test,y_pred_ada))

# Evaluate model with ROC curve
y_pred_prob = ada_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_ada)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Adda Boost')
plt.legend()

roc_auc_score(y_test, y_pred_ada)

#Applying different Models

#Support Vector Classifier
model = SVC(kernel='rbf',C=30,gamma='auto',probability=True)
model.fit(X_train, y_train)
model.score(X_train, y_train)

model.score(X_test,y_test)

pred_svc = model.predict(X_test)

print(confusion_matrix(y_test,pred_svc))
print(classification_report(y_test,pred_svc))

# Hyper Parameter Tuning
# from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(SVC(gamma='auto'), {'C': [1,10,20,30,35,40],
#                                      'kernel' : ['rbf','linear','sigmoid']},cv=5)

# clf.fit(X_train, y_train)
# print(clf.best_score_,clf.best_params_,clf.cv_results)

# Evaluate model with ROC curve
y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, pred_svc)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Support Vector')
plt.legend()

roc_auc_score(y_test, pred_svc)




#KNN
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train, y_train)
knn.score(X_train, y_train)

knn.score(X_test,y_test)

pred = knn.predict(X_test)

#Evaluate KNN  Model
print("accuracy_score for KNN Model : ",accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Choosing the good K value
accuracy_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, X,dataset['Exited'], cv=5)
    accuracy_rate.append(score.mean())

# Choosing the good K value
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, X,dataset['Exited'], cv=5)
    error_rate.append(1-score.mean())

#Check error rate in graph
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")

#accuracy error rate in graph
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title("accuracy_rate vs K Value")
plt.xlabel("K")
plt.ylabel("accuracy_rate")




#Voting Classifier
from sklearn.ensemble import VotingClassifier
volt_clf = VotingClassifier(estimators=[('gb', gb_clf),
                                        ('ada',ada_clf )],
                                           voting = 'soft')

volt_clf.fit(X_train, y_train)

volt_clf.score(X_train, y_train)

pred_volt_clf = volt_clf.predict(X_test)

#Evaluate Voting Classifier  Model
print("accuracy_score for Voting Classifier Model : ",accuracy_score(y_test,pred_volt_clf))
print(confusion_matrix(y_test,pred_volt_clf))
print(classification_report(y_test,pred_volt_clf))

y_pred_prob = volt_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, pred_volt_clf)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Voting Classifier')
plt.legend()


roc_auc_score(y_test, pred_volt_clf)



#Combining all ROC for Visualization

algos = [log_clf, tree_clf, rmd_clf, gb_clf, ada_clf, model, volt_clf]
labels = ['Logistic Regression', 'Decision Tree', 'Random forest', 'Gradient Boosting', 'Adda Boosting', 'Supports Vector', 'Voting Classifier']

plt.figure(figsize = (12,8))
plt.plot([0,1], [0,1], 'k--')

for i in range(len(algos)):
    y_pred_prob = algos[i].predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=labels[i])

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')



# ANN(Artificial Neuro Network)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding
from keras.activations import relu,sigmoid

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

#Adding Second Hidden layers
classifier.add(Dense(6,kernel_initializer='he_uniform',activation='relu'))

#Adding Output Layer
classifier.add(Dense(1,kernel_initializer='glorot_uniform',activation='sigmoid'))

#Compile ANN
classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fitting the ANN to the training set
model_history = classifier.fit(X_train, y_train,batch_size=10,epochs=100)

#Prediction on Test data set
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#Calculate the score for test data
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score

#List all data in history
print(model_history.history.keys())

#Summerise history of accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Summerise history of Loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Hyperparameter Tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            # model.add(Droupout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            # model.add(Droupout(0.3))

    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))  # No activation beyond this point

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)  # verbose=0 nothing get displayed as code is running

layers = [[20], [40,20], [45, 30,15]]
activation = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activation, batch_size= [128, 256], epochs=[10])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=2)

grid_result = grid.fit(X_train, y_train)
grid_result

#Model best result
print(grid_result.best_score_,grid_result.best_params_)

pred_y = grid.predict(X_test)
pred_y = (pred_y>0.5)
pred_y

from sklearn.metrics import confusion_matrix
cm = onfusion_matrix(y_test, pred_y)
cm

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred_y)

# Save the model
pickle.dump(volt_clf, open('bankchurn.pkl','wb'))
model_save = pickle.load(open('bankchurn.pkl','rb'))
