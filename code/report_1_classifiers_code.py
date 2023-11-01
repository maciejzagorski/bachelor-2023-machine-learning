import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def normalize(data, column):
    return (data[column] - data[column].min()) / (data[column].max() - data[column].min())


data_diabetes = pd.read_csv('../data/report_1_classifiers_data_diabetes.csv')

# data_diabetes.drop(data_diabetes[(data_diabetes.BloodPressure == 0)].index,
#                    inplace=True)

y = data_diabetes['Outcome']
x = data_diabetes.drop('Outcome', axis=1)

x_norm = x.copy()

for column in x_norm:
    x_norm[column] = normalize(x_norm, column)

results = {"Classifier": ["Naive Bayes", "k-NN, k=3", "k-NN, k=5", "k-NN, k=11", "Decision Tree",
                          "Decision Tree, class weight"]}
data_results = pd.DataFrame(results)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0, stratify=y)
x_norm_train, x_norm_test, y_norm_train, y_norm_test = train_test_split(x_norm, y, test_size=0.33, random_state=0,
                                                                        stratify=y)

print("\n* * * Naive Bayes * * *\n")

gnb = GaussianNB()

gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

data_results.loc[0, ['Score']] = [gnb.score(x_test, y_test)]
data_results.loc[0, ['False neg.']] = [confusion_matrix(y_test, y_pred)[1][0]]

print("Score: %f" % (data_results.loc[0, ['Score']]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")

print("Normalized:")

gnb.fit(x_norm_train, y_norm_train)
y_norm_pred = gnb.predict(x_norm_test)

data_results.loc[0, ['Score (norm.)']] = [(y_norm_test == y_norm_pred).sum() / x_norm_test.shape[0]]
data_results.loc[0, ['False neg. (norm.)']] = [confusion_matrix(y_norm_test, y_norm_pred)[1][0]]

print("Score: %f" % (data_results.loc[0, ['Score (norm.)']]))
print("Confusion matrix:\n", confusion_matrix(y_norm_test, y_norm_pred), "\n")

print("* * * k-NN * * *\n")

k_list = [3, 5, 11]

i = 1
for k in k_list:
    print("k-%d neighbors:\n" % (k))

    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    data_results.loc[i, ['Score']] = [knn.score(x_test, y_test)]
    data_results.loc[i, ['False neg.']] = [confusion_matrix(y_test, y_pred)[1][0]]

    print("Score: %f" % (data_results.loc[i, ['Score']]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")

    print("Normalized:")

    knn.fit(x_norm_train, y_norm_train)
    y_norm_pred = knn.predict(x_norm_test)

    data_results.loc[i, ['Score (norm.)']] = [knn.score(x_norm_test, y_norm_test)]
    data_results.loc[i, ['False neg. (norm.)']] = [confusion_matrix(y_norm_test, y_norm_pred)[1][0]]

    print("Score: %f" % (data_results.loc[i, ['Score (norm.)']]))
    print("Confusion matrix:\n", confusion_matrix(y_norm_test, y_norm_pred), "\n")

    i += 1

print("* * * Decision Tree * * *\n")

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

data_results.loc[4, ['Score']] = [tree.score(x_test, y_test)]
data_results.loc[4, ['False neg.']] = [confusion_matrix(y_test, y_pred)[1][0]]

print("Score: %f" % (data_results.loc[4, ['Score']]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")

print("Normalized:")

tree.fit(x_norm_train, y_norm_train)
y_norm_pred = tree.predict(x_norm_test)

data_results.loc[4, ['Score (norm.)']] = [tree.score(x_norm_test, y_norm_test)]
data_results.loc[4, ['False neg. (norm.)']] = [confusion_matrix(y_norm_test, y_norm_pred)[1][0]]

print("Score: %f" % (data_results.loc[4, ['Score (norm.)']]))
print("Confusion matrix:\n", confusion_matrix(y_norm_test, y_norm_pred), "\n")

print("Using class weight:\n")

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1, class_weight={0: 0.25, 1: 1})

tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

data_results.loc[5, ['Score']] = [tree.score(x_test, y_test)]
data_results.loc[5, ['False neg.']] = [confusion_matrix(y_test, y_pred)[1][0]]

print("Score: %f" % (data_results.loc[5, ['Score']]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")

print("Normalized:")

tree.fit(x_norm_train, y_norm_train)
y_norm_pred = tree.predict(x_norm_test)

data_results.loc[5, ['Score (norm.)']] = [tree.score(x_norm_test, y_norm_test)]
data_results.loc[5, ['False neg. (norm.)']] = [confusion_matrix(y_norm_test, y_norm_pred)[1][0]]

print("Score: %f" % (data_results.loc[5, ['Score (norm.)']]))
print("Confusion matrix:\n", confusion_matrix(y_norm_test, y_norm_pred), "\n")

print("* * * Results * * *\n")

print(data_results[['Classifier', 'Score', 'Score (norm.)', 'False neg.', 'False neg. (norm.)']].to_string(),
      "\n")

print("* * * Chart * * *\n")

x_axis = np.arange(len(data_results['Classifier']))
sizes = [100] * len(data_results['False neg.'])

plt.figure(figsize=(12, 6))

plt.grid(alpha=0.3, axis='y')

plt.bar(x_axis - 0.2, data_results['Score'] * 100, 0.4, label='Score', color='blue', alpha=0.5)
plt.bar(x_axis + 0.2, data_results['Score (norm.)'] * 100, 0.4, label='Score (norm.)', color='green', alpha=0.5)

plt.xticks(x_axis, data_results['Classifier'])
plt.xlabel("Classifier")

plt.ylabel("Scores [%]")
# plt.ylim(68, 78)
plt.ylim(0, 100)

plt.title("Classifiers comparison (scores)")
plt.legend()

plt.savefig('../plots/report_1_classifiers_plot_classifiers_comparison_scores_300.svg', format='svg', dpi=300)
plt.show()

plt.figure(figsize=(12, 6))

plt.grid(alpha=0.3, axis='y')

plt.bar(x_axis - 0.2, 100 - data_results['Score'] * 100, 0.4, label='Mislabeled', color='red', alpha=0.5)
plt.bar(x_axis + 0.2, 100 - data_results['Score (norm.)'] * 100, 0.4, label='Mislabeled (norm.)', color='orange',
        alpha=0.5)

plt.scatter(x_axis - 0.2, data_results['False neg.'] / len(x_test) * 100, label='False neg.', color='red', s=sizes)
plt.scatter(x_axis + 0.2, data_results['False neg. (norm.)'] / len(x_test) * 100, label='False neg. (norm.)',
            color='orange', s=sizes)

plt.xticks(x_axis, data_results['Classifier'])
plt.xlabel("Classifier")

plt.ylabel("Mislabeled [%]")
# plt.ylim(68, 78)
plt.ylim(0, 100)

plt.title("Classifiers comparison (mislabeled)")
plt.legend()

plt.savefig('../plots/report_1_classifiers_plot_classifiers_comparison_mislabeled_300.svg', format='svg', dpi=300)
plt.show()
