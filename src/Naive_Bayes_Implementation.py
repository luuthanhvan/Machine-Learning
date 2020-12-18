import pandas as pd
from pprint import pprint

'''
    Helper functions
'''
def create_age_groups(age):
    if age <= 12:
        return "Child"
    if 12 < age <= 19:
        return "Teenager"
    if age > 19:
        return "Adult"
    else:
        return "Unknown"
    
def prepare_data(df, train_set=True):
    # create new feature
    df["Age_Group"] = df.Age.apply(create_age_groups)

    # drop features that we are not going to use
    df.drop(["Name", "Age", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)

    # rename column "Parch" to "ParCh"
    df.rename({"Parch": "ParCh"}, axis=1, inplace=True)

    # rearange order of columns
    if train_set:
        df = df[["Sex", "Pclass", "Age_Group", "Embarked", "SibSp", "ParCh", "Survived"]]
    else:
        df = df[["Sex", "Pclass", "Age_Group", "Embarked", "SibSp", "ParCh"]]
    
    return df

def replace_string(df):
    df.Age_Group.replace({"Adult": 0, "Unknown": 1, "Teenager": 2, "Child": 3}, inplace=True)
    df.Embarked.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
    df.Sex.replace({"male": 0, "female": 1}, inplace=True)

    return df

def load_data():
    df_train = pd.read_csv("../data_set/Titanic_Dataset/train.csv", index_col="PassengerId")
    df_test = pd.read_csv("../data_set/Titanic_Dataset/test.csv", index_col="PassengerId")
    test_labels = pd.read_csv("../data_set/Titanic_Dataset/test_labels.csv", index_col="PassengerId", squeeze=True)

    return df_train, df_test, test_labels

'''
example_table = {
    
    "Sex": {"female": [0.15, 0.68],
            "male": [0.85, 0.32]},
    
    "Pclass": {1: [0.15, 0.40],
               2: [0.18, 0.25],
               3: [0.68, 0.35]},
    
    "class_names": [0, 1],
    "class_counts": [549, 342]
}
'''

def create_table(df, label_column):
    table = {}

    # determine values for the label
    value_counts = df[label_column].value_counts().sort_index()
    table["class_names"] = value_counts.index.to_numpy()
    table["class_counts"] = value_counts.values

    # determine probabilities for the features
    for feature in df.drop(label_column, axis=1).columns:
        table[feature] = {}

        # determine counts
        counts = df.groupby(label_column)[feature].value_counts()
        df_counts = counts.unstack(label_column)

        # add one count to avoid "problem of rare values"
        if df_counts.isna().any(axis=None):
            df_counts.fillna(value=0, inplace=True)
            df_counts += 1

        # calculate probabilities
        df_probabilities = df_counts / df_counts.sum()
        for value in df_probabilities.index:
            probabilities = df_probabilities.loc[value].to_numpy()
            table[feature][value] = probabilities
            
    return table

def predict_example(row, lookup_table):
    
    class_estimates = lookup_table["class_counts"]
    for feature in row.index:

        try:
            value = row[feature]
            probabilities = lookup_table[feature][value]
            class_estimates = class_estimates * probabilities

        # skip in case "value" only occurs in test set but not in train set
        # (i.e. "value" is not in "lookup_table")
        except KeyError:
            continue

    index_max_class = class_estimates.argmax()
    prediction = lookup_table["class_names"][index_max_class]
    
    return prediction



def main():
    df_train, df_test, test_labels = load_data()
    # print(df_train)

    # prepare data
    df_train = prepare_data(df_train)
    df_test = prepare_data(df_test, train_set=False)
    # print(df_train)

    # handle missing values in training data
    embarked_mode = df_train.Embarked.mode()[0]
    df_train["Embarked"].fillna(embarked_mode, inplace=True)
    # print(df_train["Embarked"])

    lookup_table = create_table(df_train, label_column="Survived")
    # pprint(lookup_table)

    predictions = df_test.apply(predict_example, axis=1, args=(lookup_table,))
    # predictions.head()
    predictions_correct = predictions == test_labels
    accuracy = predictions_correct.mean()
    print(f"Accuracy: {accuracy:.3f}")
    

if __name__=="__main__":
    main()