import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



data = pd.read_csv("test_data.csv")
main_attribute = ["credit_policy","purpose","int.rate","installment","log.annual.inc","dti","fico","days.with.cr.line","revol.bal","revol.util","inq.last.6mths","delinq.2yrs","pub.rec","paid_stat"]
datas = pd.DataFrame(data,columns=main_attribute)
main_data= datas[main_attribute]
real= datas["paid_stat"]
X_train, X_test, y_train, y_test = train_test_split(main_data, real, test_size=1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)