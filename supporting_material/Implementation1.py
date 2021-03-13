import pandas as pd
from sklearn.utils import shuffle
from sklearn import tree  # For our Decision Tree
import pydotplus  # To create our Decision Tree Graph
from IPython.display import Image
from sklearn.metrics import accuracy_score


def prepareTheData(df):
    df = df.astype({'oldpeak': 'int64'})
    sex_dic = {'male': 0, 'female': 1}
    df['sex'] = df['sex'].replace(sex_dic)
    cp_dic = {'none': 0, 'weak': 1, 'medium': 2, 'severe': 3}
    df['cp'] = df['cp'].replace(cp_dic)
    fbs_dic = {False: 0, True: 1}
    df['fbs'] = df['fbs'].replace(fbs_dic)
    exang_dic = {'no': 0, 'yes': 1}
    df['exang'] = df['exang'].replace(exang_dic)
    thal_dic = {'normal': 0, 'fixed_defect': 1, 'eversable_defect': 2}
    df['thal'] = df['thal'].replace(thal_dic)
    for i in range(len(df)):
        df.at[i, 'age'] = int(df.at[i, 'age'] / 10)
        df.at[i, 'trestbps'] = int(df.at[i, 'trestbps'] / 100)
        df.at[i, 'chol'] = int(df.at[i, 'chol'] / 100)
        df.at[i, 'thalach'] = int(df.at[i, 'thalach'] / 100)
    return df


dataset = pd.read_csv("dataset\heart.csv")
prepared_data = prepareTheData(dataset)
prepared_data = shuffle(prepared_data)
training_dataset = prepared_data[0:int(len(prepared_data) * 0.8)]
training_df = training_dataset[
    ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
test_dataset = prepared_data[int(len(prepared_data) * 0.8) + 1:]
# criterion = 'gini'
criterion = 'entropy'
max_depth = 2
min_samples_split = 2
clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
clf_train = clf.fit(training_df, training_dataset['target'])

# print(tree.export_graphviz(clf_train, None))

dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(training_df.columns.values),
                                class_names=['No', 'Yes'], rounded=True,
                                filled=True)  # Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes

graph = pydotplus.graph_from_dot_data(dot_data)
image = Image(graph.create_png())

prediction = clf_train.predict(test_dataset[
                                   ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                    'oldpeak', 'slope', 'ca', 'thal']])

score = accuracy_score(test_dataset['target'].to_list(), prediction)
print(score)
graph.write_png("implementation1 criterion " + criterion + ",max_depth " + str(max_depth) + ",min_samples_split" + str(
    min_samples_split) + " score " + str(score) + ".png")
