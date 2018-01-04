import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import accuracy_score

svm = SVC()
cnt = Counter()
data = pd.read_csv("test_data.csv")
main_attribute = ["credit_policy","purpose","int.rate","installment","log.annual.inc","dti","fico","days.with.cr.line","revol.bal","revol.util","inq.last.6mths","delinq.2yrs","pub.rec","paid_stat"]
datas = pd.DataFrame(data,columns=main_attribute)

for  i in datas['purpose']:
    cnt[i] += 1
k=0
t={}
for i in cnt.keys():
    t[i]=k
k+=1

datas['purpose']=datas.purpose.map(t)
main_data= datas[main_attribute]
real= datas["paid_stat"]
X_train, X_test, y_train, y_test = train_test_split(main_data, real, test_size=1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

svm.fit(X_train,y_train)
print (X_train)
predictions=svm.predict(X_test)
print( accuracy_score(y_test,predictions))