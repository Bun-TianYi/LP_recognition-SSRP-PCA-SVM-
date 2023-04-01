from sklearn.svm import SVC
import pandas as pd
from Tool.utils import *
from joblib import dump

train_data = pd.read_pickle('train_data.pkl')
test_data = pd.read_pickle('test_data.pkl')
train_label = np.array(train_data)[:, -1]
train_data = np.array(train_data)[:, 1:-1]
test_label = np.array(test_data)[:, -1]
test_data = np.array(test_data)[:, 1:-1]

# 这一段代码为训练代码，若要训练就把这段代码取消注释

clf = SVC(probability=True)
clf.fit(train_data, train_label)
dump(clf, 'SVMmodel.joblib')
# print(clf.predict_proba(train_data[0, :]))
#

# 这段代码加载已保存模型，非训练情况就不要注释该段代码
# clf = load('SVMmodel.joblib')
#
# # 该模块可以输出精度
print('训练集：', clf.score(train_data, train_label))
print('预测集：', clf.score(test_data, test_label))



