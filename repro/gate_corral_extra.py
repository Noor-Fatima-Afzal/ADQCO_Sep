import sys; sys.path.insert(0,".")
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from GSI import create_feature_map, pegasusQSVCTrain, pegasusQSVCTest
df = fetch_data('corral'); y = df['target']; X = df.drop(['target'], axis=1)
Xf, Xt, yf, yt = train_test_split(X, y, test_size=0.2, random_state=12345)
Xtr, Xv, ytr, yv = train_test_split(Xf, yf, test_size=0.25, random_state=12345)
print("n train/val/test", len(Xtr), len(Xv), len(Xt))
qc = create_feature_map(Xtr.shape[1])
m, s, t = pegasusQSVCTrain(Xtr, ytr, Xv, yv, qc)
ts, _ = pegasusQSVCTest(m, Xt, yt)
print("UNPRUNED 27-gate ZZFeatureMap: val", s, "test", ts)
for name, clf in [("rbf-svm", SVC()), ("logreg", LogisticRegression(max_iter=1000))]:
    clf.fit(Xtr, ytr); print(name, "val", clf.score(Xv, yv), "test", clf.score(Xt, yt))
