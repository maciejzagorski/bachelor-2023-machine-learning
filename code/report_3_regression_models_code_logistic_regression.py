import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

data_product_2 = pd.read_csv('../data/report_3_regression_models_product2.csv')

data_product_2.drop([data_product_2.columns[0]], axis=1, inplace=True)

data_product_2_temp = data_product_2.copy()
data_product_2_temp = data_product_2_temp[data_product_2_temp.duplicated(subset=
                                                                         ['x', 'y'],
                                                                         keep=False)]
data_product_2_temp.drop_duplicates(keep='first', inplace=True)
data_product_2_temp = data_product_2_temp[data_product_2_temp.duplicated(subset=
                                                                         ['x', 'y'],
                                                                         keep=False)]

data_product_2_clean = data_product_2.merge(data_product_2_temp, how="outer",
                                            indicator=True)
data_product_2_clean = data_product_2_clean.loc[data_product_2_clean["_merge"]
                                                == "left_only"].drop("_merge", axis=1)

print(data_product_2)
print(data_product_2_clean)

X = data_product_2.drop('good', axis=1)
X_clean = data_product_2_clean.drop('good', axis=1)

y = data_product_2['good']
y_clean = data_product_2_clean['good']

stdsc = StandardScaler()

X_std = stdsc.fit_transform(X)
X_clean_std = stdsc.fit_transform(X_clean)

f, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 6))

plt1.scatter(ma.array(X_std[:, 0], mask=y), ma.array(X_std[:, 1], mask=y),
             c='#1f77b4', marker='o', edgecolor='black', s=50, label="0")
# plt1.set(xlabel="x [standaryzowana]", ylabel="y [standaryzowana]")
plt1.set(xlabel="x [standardised]", ylabel="y [standardised]")
plt1.scatter(ma.array(X_std[:, 0], mask=np.logical_not(y)),
             ma.array(X_std[:, 1], mask=np.logical_not(y)),
             c='#ff7f0e', marker='s', edgecolor='black', s=50, label="1")
# plt1.set(xlabel="x [standaryzowana]", ylabel="y [standaryzowana]")
plt1.set(xlabel="x [standardised]", ylabel="y [standardised]")

plt1.legend(scatterpoints=1)
# plt1.set_title("Dane niezmodyfikowane", style="italic")
plt1.set_title("Unmodified data", style="italic")
plt1.grid()

plt2.scatter(ma.array(X_clean_std[:, 0], mask=y_clean),
             ma.array(X_clean_std[:, 1], mask=y_clean),
             c='#1f77b4', marker='o', edgecolor='black', s=50, label="0")
# plt2.set(xlabel="x [standaryzowana]", ylabel="y [standaryzowana]")
plt2.set(xlabel="x [standardised]", ylabel="y [standardised]")
plt2.scatter(ma.array(X_clean_std[:, 0], mask=np.logical_not(y_clean)),
             ma.array(X_clean_std[:, 1], mask=np.logical_not(y_clean)),
             c='#ff7f0e', marker='s', edgecolor='black', s=50, label="1")
# plt2.set(xlabel="x [standaryzowana]", ylabel="y [standaryzowana]")
plt2.set(xlabel="x [standardised]", ylabel="y [standardised]")

plt2.legend(scatterpoints=1)
# plt2.set_title("Dane \"wyczyszczone\"", style="italic")
plt2.set_title("\"Cleaned\" data", style="italic")
plt2.grid()

# f.suptitle("Zbiór danych", weight='bold')
f.suptitle("Dataset", weight='bold')
plt.tight_layout()

# f.savefig('../plots/report_3_regression_models_plot_product_csv_4_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_4_150_en.svg', format='svg', dpi=150)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.30,
                                                    random_state=0, stratify=y)

X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(X_clean_std,
                                                                            y_clean,
                                                                            test_size
                                                                            =0.30,
                                                                            random_state
                                                                            =0,
                                                                            stratify=
                                                                            y_clean)

logreg = LogisticRegression(C=10, random_state=0, solver='lbfgs', multi_class='ovr')


def score_logreg(X, y, estimator, estimator_name):
    y_pred = estimator.predict(X)
    print(estimator_name + "Score: %.3f" % (logreg.score(X, y)))


logreg.fit(X_train, y_train)

# score_logreg(X_train, y_train, logreg, "LogReg: dane niezmodyfikowane: zbiór danych"
#                                        "treningowych: ")
score_logreg(X_train, y_train, logreg, "LogReg: unmodified data: training dataset: ")
# score_logreg(X_test, y_test, logreg, "LogReg: dane niezmodyfikowane: zbiór danych"
#                                      "testowych: ")
score_logreg(X_test, y_test, logreg, "LogReg: unmodified data: test dataset: ")

logreg.fit(X_clean_train, y_clean_train)

# score_logreg(X_clean_train, y_clean_train, logreg, "LogReg: dane \"wyczyszczone\":"
#                                                    "zbiór danych treningowych: ")
score_logreg(X_clean_train, y_clean_train, logreg, "LogReg: \"cleaned\" data: training"
                                                   "dataset: ")
# score_logreg(X_clean_test, y_clean_test, logreg, "LogReg: dane \"wyczyszczone\":"
#                                                  "zbiór danych testowych: ")
score_logreg(X_clean_test, y_clean_test, logreg, "LogReg: \"cleaned\" data: test "
                                                 "dataset: ")


f, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 6))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

X_clean_combined = np.vstack((X_clean_train, X_clean_test))
y_clean_combined = np.hstack((y_clean_train, y_clean_test))

plot_decision_regions(X_combined, y_combined, clf=logreg, ax=plt1, markers='os')
# plt1.set(xlabel="x [standaryzowana]", ylabel="y [standaryzowana]")
plt1.set(xlabel="x [standardised]", ylabel="y [standardised]")
# plt1.set_title("Dane niezmodyfikowane (\"x\" i \"y\")", style="italic")
plt1.set_title("Unmodified data (\"x\" i \"y\")", style="italic")
plt1.grid()

plot_decision_regions(X_clean_combined, y_clean_combined, clf=logreg, ax=plt2, markers='os')
# plt2.set(xlabel="x [standaryzowana]", ylabel="y [standaryzowana]")
plt2.set(xlabel="x [standardised]", ylabel="y [standardised]")
# plt2.set_title("Dane \"wyczyszczone\" (\"x\" i \"y\")", style="italic")
plt2.set_title("\"Cleaned\" data (\"x\" i \"y\")", style="italic")
plt2.grid()

# f.suptitle("Zastosowanie regresji logistycznej", weight='bold')
f.suptitle("Application of logistic regression ", weight='bold')
plt.tight_layout()
# f.savefig('../plots/report_3_regression_models_plot_product_csv_5_1_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_5_1_150_en.svg', format='svg', dpi=150)
plt.show()

X_train = np.delete(X_train, 0, 1)
X_test = np.delete(X_test, 0, 1)
logreg.fit(X_train, y_train)

# score_logreg(X_train, y_train, logreg, "LogReg: dane niezmodyfikowane (bez \"y\"):"
#                                        "zbiór danych treningowych: ")
score_logreg(X_train, y_train, logreg, "LogReg: unmodified data (without \"y\"): "
                                       "training dataset: ")
# score_logreg(X_test, y_test, logreg, "LogReg: dane niezmodyfikowane(bez \"y\"):"
#                                      "zbiór danych testowych: ")
score_logreg(X_test, y_test, logreg, "LogReg: unmodified data (without \"y\"): "
                                     "test dataset: ")

X_clean_train = np.delete(X_clean_train, 0, 1)
X_clean_test = np.delete(X_clean_test, 0, 1)
logreg.fit(X_clean_train, y_clean_train)

# score_logreg(X_clean_train, y_clean_train, logreg, "LogReg: dane \"wyczyszczone\" (bez"
#                                                    "\"y\"): zbiór danych treningowych: ")
score_logreg(X_clean_train, y_clean_train, logreg, "LogReg: \"cleaned\" data: (without "
                                                   "\"y\"): training dataset: ")
# score_logreg(X_clean_test, y_clean_test, logreg, "LogReg: dane \"wyczyszczone\" (bez"
#                                                  "\"y\"): zbiór danych testowych: ")
score_logreg(X_clean_test, y_clean_test, logreg, "LogReg: \"cleaned\" data: (without "
                                                 "\"y\"): test dataset: ")

f, (plt3, plt4) = plt.subplots(1, 2, figsize=(12, 6))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

X_clean_combined = np.vstack((X_clean_train, X_clean_test))
y_clean_combined = np.hstack((y_clean_train, y_clean_test))

plot_decision_regions(X_combined, y_combined, clf=logreg, ax=plt3, markers='os')
# plt3.set(xlabel="x [standaryzowana]")
plt3.set(xlabel="x [standardised]")
# plt3.set_title("Dane niezmodyfikowane (\"x\")", style="italic")
plt3.set_title("Unmodified data (\"x\")", style="italic")
plt3.grid()

plot_decision_regions(X_clean_combined, y_clean_combined, clf=logreg, ax=plt4, markers='os')
# plt4.set(xlabel="x [standaryzowana]")
plt4.set(xlabel="x [standardised]")
# plt4.set_title("Dane \"wyczyszczone\" (\"x\")", style="italic")
plt4.set_title("\"Cleaned\" data (\"x\")", style="italic")
plt4.grid()

# f.suptitle("Zastosowanie regresji logistycznej", weight='bold')
f.suptitle("Application of logistic regression", weight='bold')
plt.tight_layout()
# f.savefig('../plots/report_3_regression_models_plot_product_csv_5_2_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_5_2_150_en.svg', format='svg', dpi=150)
plt.show()
