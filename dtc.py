from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# アヤメの訓練データ
iris = load_iris()

x = iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(x, y)

# save model
joblib.dump(tree_clf, "trained_model.bin")

from graphviz import Source
from sklearn.tree import export_graphviz
import os

export_graphviz(
    tree_clf,
    out_file=os.path.join("iris_tree.dot"),
    class_names=iris.target_names,
    rounded=True,
    filled=True
)