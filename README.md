_[(English version below)](#machine-learning)_

# Uczenie maszynowe

_(Raporty przygotowane w 2023 r. w trakcie kursu z zakresu uczenia maszynowego, w ramach studiów inżynierskich na
kierunku informatyka na Polsko-Japońskiej Akademii Technik Komputerowych w Gdańsku)._

1. [Sprawdzenie działania wybranych klasyfikatorów](#raport-z-projektu-1)
2. [Sprawdzenie działania i weryfikacja poprawności wybranych algorytmów grupujących](#raport-z-projektu-2)
3. [Sprawdzenie działania modeli regresywnych](#raport-z-projektu-3)

## Raport z projektu 1

### Sprawdzenie działania wybranych klasyfikatorów

Przedmiotem projektu było sprawdzenie, jak działają następujące, poznane klasyfikatory (metody, algorytmy uczenia
maszynowego (nadzorowanego)):

* naiwny klasyfikator Bayesa (_Naive Bayes_),
* klasyfikator k-najbliższych sąsiadów (_k-Nearest Neighbors_, k-NN) – z różnym parametrem k:
    * k = 3
    * k = 5
    * k = 11
* algorytm wykorzystujący drzewa decyzyjne (_Decision tree_).

Sprawdzenia działania powyższych klasyfikatorów dokonano w oparciu o przykładowy zbiór danych „diabetes.csv”,
zawierający informacje o kobietach indiańskiego pochodzenia z USA (ich parametry medyczne), które zachorowały lub nie
zachorowały na cukrzycę. Zastosowanie klasyfikatora w takim wypadku służy diagnozowaniu choroby na podstawie znanych
danych medycznych.

## Raport z projektu 2

### Sprawdzenie działania i weryfikacja poprawności wybranych algorytmów grupujących

Przedmiotem projektu było sprawdzenie, jak działają następujące, poznane algorytmy grupujące:

* algorytm centroidów (k-średnich, _k-means_),
* algorytm gęstościowej klasteryzacji przestrzennej z uwzględnieniem szumu (_density based spatial clustering of
  applicaions with noise_, DBSCAN).

Dodatkowo wykonano prostą implementację algorytmu hierarchicznego – klasteryzacji aglomeracyjnej (_agglomerative
clustering_).

Ponadto celem realizacji projektu była weryfikacja poprawności (rezultatów zastosowania) przedmiotowych algorytmów.

Powyższego dokonano w oparciu o przykładowe zbiory danych „iris2D.csv” oraz „irisORG.csv”, oba zawierające dane o
kwiatach należących do rodzaju irysów. Zastosowanie algorytmów grupujących ma w tym przypadku na celu połączenie kwiatów
w grupy – ich podział na gatunki, tj. przyporządkowanie poszczególnych kwiatów do kolejno wyodrębnianych grup (
odpowiadających gatunkom kwiatów z rodzaju irysów).

## Raport z projektu 3

### Sprawdzenie działania modeli regresywnych

Przedmiotem projektu było sprawdzenie, jak działają modele regresywne uczenia maszynowego:

* model regresji liniowej,
* model regresji wielomianowej drugiego stopnia (dopasowanie wielomianu drugiego stopnia),
* model regresji wielomianowej trzeciego stopnia.

Ponadto dokonano sprawdzenia działania klasyfikacji dokonywanej przy pomocy regresji logistycznej (będącej właśnie
algorytmem klasyfikacji).

Powyższego dokonano w oparciu o przykładowe zbiory danych „product.csv” oraz „product2.csv”, zawierające wartości
współczynników (cech, parametrów) „x” oraz „y”. Zastosowanie modeli regresywnych w tym wypadku pozwala na
przewidywanie (określenie), jaka powinna być wartość współczynnika „y” w przypadku produktu o współczynniku „x”.
Natomiast wykorzystanie modelu regresji logistycznej służy w tym wypadku kwalifikowaniu próbek o danych parametrach do
określonej kategorii.

# Machine Learning

_(The reports prepared in 2023 during the machine learning course, as a part of the bachelor's degree program in
computer science at Polsko-Japońska Akademia Technik Komputerowych, Gdańsk._

_The reports, originally wirtten in Polish, were translated to English using machine translation methods)._

1. [Evaluating the performance of selected classifiers](#project-report-1)
2. [Assessing the operation and verifying the accuracy of selected clustering algorithms](#project-report-2)
3. [Evaluating the performance of regression models](#project-report-3)

## Project report 1

### Evaluating the performance of selected classifiers

The object of the project was to assess how the following learned classifiers (methods, machine learning (supervised)
algorithms) work:

* Naive Bayes classifier,
* k-Nearest Neighbours (k-NN) classifier – with different parameter k:
    * k = 3
    * k = 5
    * k = 11
* decision tree algorithm.

The performance of the above classifiers was evaluated using the sample dataset “diabetes.csv”, which contains
information about women of Native American origin from the USA (their medical parameters) who did or did not develop
diabetes. The use of a classifier in this context serves to diagnose the disease based on known medical data.

## Project report 2

### Assessing the operation and verifying the accuracy of selected clustering algorithms

The object of the project was to evaluate the performance of the following learned clustering algorithms:

* centroid algorithm (k-means),
* density based spatial clustering of applicaions with noise (DBSCAN) algorithm.

Additionally, a basic implementation of the hierarchical clustering algorithm (agglomerative clustering) was provided.

Furthermore, the object of the project was to verify the accuracy of the algorithms in question (results of their
application).

The above was done on the basis of the sample datasets “iris2D.csv” and “irisORG.csv”, both of which contain data on
flowers belonging to the genus iris. The application of grouping algorithms in this case is aimed at combining flowers
into groups – their division into species, i.e. assigning individual flowers to successively extracted groups (
corresponding to flower species of the iris genus).

## Project report 3

### Evaluating the performance of regression models

The object of the project was to assess how the machine learning regression models work:

* linear regression model,
* second-degree polynomial regression model (second-degree polynomial fitting),
* third-degree polynomial regression model.

Additionally, an assessment was made on the performance of the classification performed using logistic regression (which
is precisely a classification algorithm (a classifier)).

The above was done on the basis of the sample datasets “product.csv” and “product2.csv”, which contain values of “x”
and “y” coefficients (features, parameters). The use of regression models in this context makes it possible to predict
(to determine) what the value of the “y” coefficient should be for a product with a given “x” coefficient. On the other
hand, the use of a logistic regression model in this context serves to qualify samples with given parameters into a
specific category.