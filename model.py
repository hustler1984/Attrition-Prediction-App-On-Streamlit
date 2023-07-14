#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pickle import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def open_data(path="data/HR Employee Attrition.csv"):
    df = pd.read_csv(path)
    df = df[['Attrition','MonthlyIncome','OverTime','EnvironmentSatisfaction','TotalWorkingYears','NumCompaniesWorked']]
    
    return df

def split_data(df: pd.DataFrame):
    category_map = {'Yes': 1,'No': 0}
    y = df['Attrition'].map(category_map)
    X = df[['MonthlyIncome','OverTime','EnvironmentSatisfaction','TotalWorkingYears','NumCompaniesWorked']]

    return X, y

def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    categorical_column2 = ['OverTime']
    X_df = pd.get_dummies(X_df, columns = categorical_column2 ,
                      prefix_sep = "_",drop_first = False)[['MonthlyIncome','EnvironmentSatisfaction','TotalWorkingYears',
                                                            'NumCompaniesWorked','OverTime_No']]
    categorical = ['OverTime_No']
    numeric_features = [col for col in X_df.columns if col not in categorical]

    column_transformer = ColumnTransformer([('pass', 'passthrough', categorical),
                                            ('scaling', StandardScaler(), numeric_features)])

    X_df_transformed = column_transformer.fit_transform(X_df)

    lst = list(categorical)
    lst.extend(column_transformer.transformers_[1][1].get_feature_names_out())

    X_df = pd.DataFrame(X_df_transformed, columns=lst)[['MonthlyIncome','EnvironmentSatisfaction','TotalWorkingYears',
                                                        'NumCompaniesWorked','OverTime_No']]

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    decision_tree_parameters = {'criterion': 'entropy', 'max_depth': 9, 'max_features': 0.33334, 'max_leaf_nodes': 20, 'min_samples_leaf': 15, 'min_samples_split': 2, 'random_state': 0}
    model = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.01, estimator=DecisionTreeClassifier(**decision_tree_parameters),random_state=0)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Сотрудник останется с вероятностью",
        1: "Сотрудник уволится с вероятностью"
    }

    encode_prediction = {
        0: "Сотрудник ещё поработает",
        1: "Скоро сотрудник уволится"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)




