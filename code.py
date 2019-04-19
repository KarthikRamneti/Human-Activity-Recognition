import os
#print(os.listdir("C:\Users\Karthik\Desktop\New folder"))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


data = pd.read_csv(r"C:\Users\Karthik\Desktop\New folder\X_train.txt", sep = " ", header=None )
datay = pd.read_csv(r"C:\Users\Karthik\Desktop\New folder\Y_train.txt", sep = " ", header=None )
x_train = pd.read_csv(r"C:\Users\Karthik\Desktop\New folder\X_test.txt", sep = " ", header=None )

X_train, X_test, y_train, y_test = train_test_split(data, datay, test_size=0.3, random_state=1, stratify=datay)

sc = StandardScaler()
sc.fit(data)
X_train_fit = sc.transform(X_train)
X_test_fit = sc.transform(X_test)
final_fit = sc.transform(x_train)

#X_train_fit = X_train_std.values
y_train_fit = y_train.values.ravel()

#X_test_fit = X_test.values
y_test_fit = y_test.values.ravel()

rf=RandomForestClassifier(random_state=1, n_estimators=3000)
et = ExtraTreesClassifier(random_state =1, n_estimators =2000)
ada = AdaBoostClassifier(random_state =1,n_estimators =2000)
gb = GradientBoostingClassifier(random_state =1, n_estimators=2000)
#svm = SVC(random_state =1)
knn = KNeighborsClassifier()
#sgd = SGDClassifier(random_state =1)
#mlp = MLPClassifier(random_state =1)


main_model = VotingClassifier(estimators=[('rf', rf), ('et', et), ('ada', ada), ('gb', gb), ('knn', knn)], voting='soft')
main_model.fit(X_train_fit, y_train_fit)
print(main_model.score(X_test_fit,y_test_fit))

predictions = main_model.predict(final_fit)
Submission = pd.DataFrame({ "Predicted Label": predictions })
Submission.to_csv(r"C:\Users\Karthik\Desktop\New folder\submit.csv")