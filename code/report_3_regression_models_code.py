import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

data_product = pd.read_csv('../data/report_3_regression_models_product.csv')

data_product.drop([data_product.columns[0]], axis=1, inplace=True)

stdsc = StandardScaler()

data_product_stand = stdsc.fit_transform(data_product)


def remove_outliers(dataset):
    q1 = np.quantile(dataset[:, 1], 0.25)
    q3 = np.quantile(dataset[:, 1], 0.75)
    iqr = q3 - q1

    print("Q1: %.3f\nQ2: %.3f\nIQR: %.3f" % (q1, q3, iqr))

    inliers = dataset[dataset[:, 1] < q3 + 1.5 * iqr]
    inliers = inliers[inliers[:, 1] > q1 - 1.5 * iqr]

    outliers = dataset[dataset[:, 1] >= q3 + 1.5 * iqr]
    outliers = np.vstack((outliers, dataset[dataset[:, 1] <= q1 - 1.5 * iqr]))

    return inliers, outliers


data_product_stand_inliers, data_product_stand_outliers = remove_outliers(data_product_stand)

f, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 6))

plt1.scatter(data_product.x, data_product.y, c='white', marker='o',
             edgecolor='black', s=50)
plt1.set(xlabel="x", ylabel="y")
plt1.grid()

plt2.scatter(stdsc.inverse_transform(data_product_stand_inliers)[:, 0],
             stdsc.inverse_transform(data_product_stand_inliers)[:, 1], c='green',
             # marker='o', edgecolor='black', s=50, label="Próbki nieodstające (inliers)")
             marker='o', edgecolor='black', s=50, label="Inliers")
plt2.scatter(stdsc.inverse_transform(data_product_stand_outliers)[:, 0],
             stdsc.inverse_transform(data_product_stand_outliers)[:, 1], c='red',
             # marker='o', edgecolor='black', s=50, label="Próbki odstające (outliers)")
             marker='o', edgecolor='black', s=50, label="Outliers")
plt2.legend(scatterpoints=1)
plt2.set(xlabel="x", ylabel="y")
plt2.grid()

# f.suptitle("Zbiór danych", weight='bold')
f.suptitle("Dataset", weight='bold')
plt.tight_layout()

# f.savefig('../plots/report_3_regression_models_plot_product_csv_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_150_en.svg', format='svg', dpi=150)
plt.show()

X_std = data_product_stand_inliers[:, 0, np.newaxis]
y_std = data_product_stand_inliers[:, 1, np.newaxis]
X_fit = np.arange(X_std.min(), X_std.max(), 0.1)[:, np.newaxis]
# X_fit = np.sort(X_std.flatten())[:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.30,
                                                    random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

ransac = RANSACRegressor(estimator=LinearRegression(), max_trials=100,
                         loss='absolute_error', random_state=0)
ransac.fit(X_train, y_train)

quadratic = PolynomialFeatures(degree=2)
X_train_quad = quadratic.fit_transform(X_train)
X_test_quad = quadratic.fit_transform(X_test)
lr_quadratic = LinearRegression()
lr_quadratic.fit(X_train_quad, y_train)

cubic = PolynomialFeatures(degree=3)
X_train_cubic = cubic.fit_transform(X_train)
X_test_cubic = cubic.fit_transform(X_test)

lr_cubic = LinearRegression()
lr_cubic.fit(X_train_cubic, y_train)


def score(X, y, estimator, estimator_name):
    y_pred = estimator.predict(X)
    print(estimator_name + '\nMSE: %.3f,\nR^2: %.3f\n' % (mean_squared_error(y, y_pred),
                                                          r2_score(y, y_pred)))


# score(X_train, y_train, lr, "LR: zbiór danych treningowych:")
score(X_train, y_train, lr, "LR: training dataset:")
# score(X_test, y_test, lr, "LR: zbiór danych testowych:")
score(X_test, y_test, lr, "LR: test dataset:")

# score(X_train, y_train, ransac, "RANSAC: zbiór danych treningowych:")
score(X_train, y_train, ransac, "RANSAC: training dataset:")
# score(X_test, y_test, ransac, "RANSAC: zbiór danych testowych:")
score(X_test, y_test, ransac, "RANSAC: test dataset:")

# score(X_train_quad, y_train, lr_quadratic, "^2: zbiór danych treningowych:")
score(X_train_quad, y_train, lr_quadratic, "^2: training dataset:")
# score(X_test_quad, y_test, lr_quadratic, "^2: zbiór danych testowych:")
score(X_test_quad, y_test, lr_quadratic, "^2: test dataset:")

# score(X_train_cubic, y_train, lr_cubic, "^3: zbiór danych treningowych:")
score(X_train_cubic, y_train, lr_cubic, "^3: training dataset:")
# score(X_test_cubic, y_test, lr_cubic, "^3: zbiór danych testowych:")
score(X_test_cubic, y_test, lr_cubic, "^3: test dataset:")

f, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 6))

plt1.scatter(stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 0],
             stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 1], c='blue', marker='o',
             # edgecolor='black', s=50, label="Dane treningowe")
             edgecolor='black', s=50, label="Training data")
plt1.scatter(stdsc.inverse_transform(np.hstack((X_test, y_test)))[:, 0],
             stdsc.inverse_transform(np.hstack((X_test, y_test)))[:, 1], c='gold', marker='o',
             # edgecolor='black', s=50, label="Dane testowe")
             edgecolor='black', s=50, label="Test data")
plt1.scatter(stdsc.inverse_transform(data_product_stand_outliers)[:, 0],
             stdsc.inverse_transform(data_product_stand_outliers)[:, 1], c='red',
             # marker='o', edgecolor='black', s=50, label="Próbki odstające (outliers)")
             marker='o', edgecolor='black', s=50, label="Outliers")
plt1.legend(scatterpoints=1)
plt1.set(xlabel="x", ylabel="y")
# plt1.set_title("Z uwględnieniem próbek odstających", style="italic")
plt1.set_title("Including outliers", style="italic")
plt1.grid()

plt2.scatter(stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 0],
             stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 1], c='blue', marker='o',
             # edgecolor='black', s=50, label="Dane treningowe")
             edgecolor='black', s=50, label="Training data")
plt2.scatter(stdsc.inverse_transform(np.hstack((X_test, y_test)))[:, 0],
             stdsc.inverse_transform(np.hstack((X_test, y_test)))[:, 1], c='gold', marker='o',
             # edgecolor='black', s=50, label="Dane testowe")
             edgecolor='black', s=50, label="Test data")
plt2.legend(scatterpoints=1)
plt2.set(xlabel="x", ylabel="y")
# plt2.set_title("Bez uwzględnienia próbek odstających", style="italic")
plt2.set_title("Excluding outliers", style="italic")
plt2.grid()

# f.suptitle("Podział zbiorów danych", weight='bold')
f.suptitle("Split of datasets", weight='bold')
plt.tight_layout()
# f.savefig('../plots/report_3_regression_models_plot_product_csv_2_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_2_150_en.svg', format='svg', dpi=150)
plt.show()

f, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 6))

y_lr = lr.predict(X_fit)
# y_lr_label = "Regresja liniowa: y = (%.3f)*x + (%.3f)" % (lr.coef_[0, 0], lr.intercept_[0])
y_lr_label = "Linear regression: y = (%.3f)*x + (%.3f)" % (lr.coef_[0, 0], lr.intercept_[0])

y_ransac = ransac.predict(X_fit)
y_ransac_label = "RANSAC: y = (%.3f)*x + (%.3f)" % (ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

y_quadratic = lr_quadratic.predict(quadratic.fit_transform(X_fit))
# y_quadratic_label = "Regresja wielomianowa (^2): y =\n(%.3f)*x^2 + %.3f*x + (%.3f)" % (
#     lr_quadratic.coef_[0, 2], lr_quadratic.coef_[0, 1], lr_quadratic.intercept_[0])
y_quadratic_label = "Polynomial regression (^2): y =\n(%.3f)*x^2 + %.3f*x + (%.3f)" % (
    lr_quadratic.coef_[0, 2], lr_quadratic.coef_[0, 1], lr_quadratic.intercept_[0])

# print(lr_quadratic.coef_)
# print(lr_quadratic.intercept_)
# print(lr_quadratic.predict(np.array([1, -0.5, 0.25]).reshape(1, -1)))
#
# print(lr_cubic.coef_)
# print(lr_cubic.intercept_)
# print(lr_cubic.predict(np.array([1, 1.5, 2.25, 3.375]).reshape(1, -1)))

y_cubic = lr_cubic.predict(cubic.fit_transform(X_fit))
# y_cubic_label = "Regresja wielomianowa (^3): y =\n%.3f*x^3 + %.3f*x^2 + (%.3f)*x + (%.3f)" % (
#     lr_cubic.coef_[0, 3], lr_cubic.coef_[0, 2], lr_cubic.coef_[0, 1], lr_cubic.intercept_[0])
y_cubic_label = "Polynomial regression (^3): y =\n%.3f*x^3 + %.3f*x^2 + (%.3f)*x + (%.3f)" % (
    lr_cubic.coef_[0, 3], lr_cubic.coef_[0, 2], lr_cubic.coef_[0, 1], lr_cubic.intercept_[0])

plt1.scatter(X_train, y_train, c='lightgrey', marker='o', s=50)
plt1.scatter(X_test, y_test, c='lightgrey', marker='o', s=50)

plt1.plot(X_fit, y_lr, color='blue', label=y_lr_label, linewidth=2)
plt1.plot(X_fit, y_ransac, color='red', label=y_ransac_label, linewidth=2)
plt1.plot(X_fit, y_quadratic, color='green', label=y_quadratic_label, linewidth=2)
plt1.plot(X_fit, y_cubic, color='orange', label=y_cubic_label, linewidth=2)

plt1.legend(scatterpoints=1)
plt1.set(xlabel="x", ylabel="y")
# plt1.set_title("Wartości ustandaryzowane", style="italic")
plt1.set_title("Standardised values", style="italic")
plt1.grid()

lr.fit(stdsc.inverse_transform(np.hstack((X_fit, y_lr)))[:, 0].reshape(-1, 1),
       stdsc.inverse_transform(np.hstack((X_fit, y_lr)))[:, 1].reshape(-1, 1))

ransac.fit(stdsc.inverse_transform(np.hstack((X_fit, y_ransac)))[:, 0].reshape(-1, 1),
           stdsc.inverse_transform(np.hstack((X_fit, y_ransac)))[:, 1].reshape(-1, 1))

lr_quadratic.fit(quadratic.fit_transform(stdsc.inverse_transform(np.hstack((X_fit, y_quadratic)))[:, 0].reshape(-1, 1)),
                 stdsc.inverse_transform(np.hstack((X_fit, y_quadratic)))[:, 1].reshape(-1, 1))

lr_cubic.fit(cubic.fit_transform(stdsc.inverse_transform(np.hstack((X_fit, y_cubic)))[:, 0].reshape(-1, 1)),
             stdsc.inverse_transform(np.hstack((X_fit, y_cubic)))[:, 1].reshape(-1, 1))

# y_lr_label = "Regresja liniowa: y = (%.3f)*x + (%.3f)" % (lr.coef_[0, 0], lr.intercept_[0])
y_lr_label = "Linear regression: y = (%.3f)*x + (%.3f)" % (lr.coef_[0, 0], lr.intercept_[0])
y_ransac_label = "RANSAC: y = (%.3f)*x + (%.3f)" % (ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])
# y_quadratic_label = "Regresja wielomianowa (^2): y =\n%.3f*x^2 + (%.3f)*x + %.3f" % (
#     lr_quadratic.coef_[0, 2], lr_quadratic.coef_[0, 1], lr_quadratic.intercept_[0])
y_quadratic_label = "Polynomial regression (^2): y =\n%.3f*x^2 + (%.3f)*x + %.3f" % (
    lr_quadratic.coef_[0, 2], lr_quadratic.coef_[0, 1], lr_quadratic.intercept_[0])
# y_cubic_label = "Regresja wielomianowa (^3): y =\n%.3f*x^3 + (%.3f)*x^2 + %.3f*x + (%.3f)" % (
#     lr_cubic.coef_[0, 3], lr_cubic.coef_[0, 2], lr_cubic.coef_[0, 1], lr_cubic.intercept_[0])
y_cubic_label = "Polynomial regression (^3): y =\n%.3f*x^3 + (%.3f)*x^2 + %.3f*x + (%.3f)" % (
    lr_cubic.coef_[0, 3], lr_cubic.coef_[0, 2], lr_cubic.coef_[0, 1], lr_cubic.intercept_[0])

plt2.scatter(stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 0],
             stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 1], c='lightgrey', marker='o',
             s=50)
plt2.scatter(stdsc.inverse_transform(np.hstack((X_test, y_test)))[:, 0],
             stdsc.inverse_transform(np.hstack((X_test, y_test)))[:, 1], c='lightgrey', marker='o',
             s=50)

plt2.plot(stdsc.inverse_transform(np.hstack((X_fit, y_lr)))[:, 0],
          stdsc.inverse_transform(np.hstack((X_fit, y_lr)))[:, 1], color='blue', label=y_lr_label, linewidth=2)
plt2.plot(stdsc.inverse_transform(np.hstack((X_fit, y_ransac)))[:, 0],
          stdsc.inverse_transform(np.hstack((X_fit, y_ransac)))[:, 1], color='red', label=y_ransac_label, linewidth=2)
plt2.plot(stdsc.inverse_transform(np.hstack((X_fit, y_quadratic)))[:, 0],
          stdsc.inverse_transform(np.hstack((X_fit, y_quadratic)))[:, 1], color='green',
          label=y_quadratic_label, linewidth=2)
plt2.plot(stdsc.inverse_transform(np.hstack((X_fit, y_cubic)))[:, 0],
          stdsc.inverse_transform(np.hstack((X_fit, y_cubic)))[:, 1], color='orange',
          label=y_cubic_label,
          linewidth=2)

plt2.legend(scatterpoints=1)
plt2.set(xlabel="x", ylabel="y")
# plt2.set_title("Wartości rzeczywiste", style="italic")
plt2.set_title("Real values", style="italic")
plt2.grid()

# f.suptitle("Linie regresji", weight='bold')
f.suptitle("Regression lines", weight='bold')
plt.tight_layout()
# f.savefig('../plots/report_3_regression_models_plot_product_csv_3_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_3_150_en.svg', format='svg', dpi=150)
plt.show()

ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

f, (plt1) = plt.subplots(1, 1, figsize=(6, 6))

plt1.scatter(stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 0][inlier_mask],
             stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 1][inlier_mask], c='cyan', marker='o',
             # edgecolor='black', s=50, label="Próbki nieodstające (inliers) wg RANSAC")
             edgecolor='black', s=50, label="Inliers according to RANSAC")
plt1.scatter(stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 0][outlier_mask],
             stdsc.inverse_transform(np.hstack((X_train, y_train)))[:, 1][outlier_mask], c='pink', marker='o',
             # edgecolor='black', s=50, label="Próbki odstające (outliers) wg RANSAC")
             edgecolor='black', s=50, label="Outliers according to RANSAC")

plt1.legend(scatterpoints=1)
plt1.set(xlabel="x", ylabel="y")
# plt1.set_title("Próbki nieodstające i odstające (dane treningowe)", style="italic")
plt1.set_title("Inliers and outliers (training data)", style="italic")
plt1.grid()

# f.suptitle("Regresja liniowa (RANSAC)", weight='bold')
f.suptitle("Linear regression (RANSAC)", weight='bold')
plt.tight_layout()
# f.savefig('../plots/report_3_regression_models_plot_product_csv_6_150.svg', format='svg', dpi=150)
f.savefig('../plots/report_3_regression_models_plot_product_csv_6_150_en.svg', format='svg', dpi=150)
plt.show()
