import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

svm = SVC()
cnt = Counter()
data = pd.read_csv("test_data.csv")
main_attribute = ["credit_policy", "purpose","int.rate", "installment", "log.annual.inc", "dti","fico", "days.with.cr.line", "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec", "paid_stat"]
datas = pd.DataFrame(data, columns=main_attribute)
np_array = datas.as_matrix(columns=main_attribute)
np_array = datas.values
kf = KFold(n_splits=2)
kf.get_n_splits(datas)
print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kf.split(np_array):
    print("Train:", train_index)
    print("Test:",test_index)
    X_train, X_test = np_array[train_index], np_array[test_index]
    y_train, y_test = np_array[train_index], np_array[test_index]
for i in datas['purpose']:
    cnt[i] += 1
k = 0
t = {}
for i in cnt.keys():
    t[i] = k
k += 1

datas['purpose'] = datas.purpose.map(t)
main_data = datas[main_attribute]
real = datas["paid_stat"]
X_train, X_test, y_train, y_test = train_test_split(main_data, real, test_size=0.5)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

svm.fit(X_train, y_train)
print(X_train)
predictions = svm.predict(X_test)
print(accuracy_score(predictions, y_test))

predict_status = datas.loc[datas['purpose'] == t['credit_card']]

predict_status1 = predict_status.loc[predict_status['paid_stat'] == 1]
predict_status2 = predict_status.loc[predict_status['paid_stat'] == 0]


predict_status.hist()
plt.show()