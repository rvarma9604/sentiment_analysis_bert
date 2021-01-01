import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import preprocessing
import argparse

def read_file(f_name):
    dataset = pd.read_csv(f_name)
    dataset.dropna(inplace=True)
    return dataset

def vector_labels(train, test):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True).fit(train['text'])
    train_vectors = vectorizer.transform(train['text'])
    test_vectors = vectorizer.transform(test['text'])

    le = preprocessing.LabelEncoder().fit(train.values[:,-1])
    train_labels = le.transform(train.values[:,-1])
    test_labels = le.transform(test.values[:,-1])

    return train_vectors, test_vectors, train_labels, test_labels

def SVM(train, test, out_dir):
    train_vectors, test_vectors, train_labels, test_labels = vector_labels(train, test)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    model = SVC()

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['auto', 'scale'],
        'degree': [4, 5, 10]
    }

    grid_svm = GridSearchCV(model, 
                            param_grid=param_grid, 
                            cv=kfolds,
                            scoring='accuracy',
                            verbose=1,
                            n_jobs=-1)

    grid_svm.fit(train_vectors, train_labels)
    print("SVM")
    print("\tBest parameters\n\t\t",grid_svm.best_params_)

    model = grid_svm.best_estimator_
    model.fit(train_vectors, train_labels)
    print("\tTrain Report:\n", classification_report(train_labels, model.predict(train_vectors)))
    print("\tTest Report:\n", classification_report(test_labels, model.predict(test_vectors)))

    with open(out_dir + '/svm.pkl', 'wb') as f:
        pickle.dump(model, f)

def LogReg(train, test, out_dir):
    train_vectors, test_vectors, train_labels, test_labels = vector_labels(train, test)

    model = LogisticRegression(solver='lbfgs').fit(train_vectors, train_labels)
    print("LogisticRegression")
    print("\tTrain Report:\n", classification_report(train_labels, model.predict(train_vectors)))
    print("\tTest Report:\n", classification_report(test_labels, model.predict(test_vectors)))

    with open(out_dir + '/logreg.pkl', 'wb') as f:
        pickle.dump(model, f)

def RandomForest(train, test, out_dir):
    train_vectors, test_vectors, train_labels, test_labels = vector_labels(train, test)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    model = RandomForestClassifier()

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20]
    }

    grid_rf = GridSearchCV(model,
                           param_grid=param_grid,
                           cv=kfolds,
                           scoring='accuracy',
                           verbose=1,
                           n_jobs=-1)

    grid_rf.fit(train_vectors, train_labels)
    print("RandomForestClassifier")
    print("\tBest parameters\n\t\t",grid_rf.best_params_)

    model = grid_rf.best_estimator_
    model.fit(train_vectors, train_labels)
    print("\tTrain Report:\n", classification_report(train_labels, model.predict(train_vectors)))
    print("\tTest Report:\n", classification_report(test_labels, model.predict(test_vectors)))

    with open(out_dir + '/rf.pkl', 'wb') as f:
        pickle.dump(model, f)

def GNB(train, test, out_dir):
    train_vectors, test_vectors, train_labels, test_labels = vector_labels(train, test)

    model = GaussianNB()
    model.fit(train_vectors.toarray(), train_labels)
    print("GaussianNB")
    print("\tTrain Report:\n", classification_report(train_labels, model.predict(train_vectors.toarray())))
    print("\tTest Report:\n", classification_report(test_labels, model.predict(test_vectors.toarray())))

    with open(out_dir + '/gnb.pkl', 'wb') as f:
        pickle.dump(model, f)

def main():
    parser = argparse.ArgumentParser(description='Compare Scores with predefined models', add_help=True)
    parser.add_argument('train', help='train csv file')
    parser.add_argument('test', help='test csv file')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--stage', default=0, type=int, help='resume analysis stage')
    args = parser.parse_args()
    print(args)

    # load dataset
    train = read_file(args.train)
    test = read_file(args.test)

    # SVM
    if args.stage <= 0:
        SVM(train, test, args.out_dir)

    # Logistic Regression
    if args.stage <= 1:
        LogReg(train, test, args.out_dir)

    # Random Forest
    if args.stage <= 2:
        RandomForest(train, test, args.out_dir)

    # Gaussian Naive Bayes
    if args.stage <= 3:
        GNB(train, test, args.out_dir)

    print('Finished')

if __name__=='__main__':
    main()
