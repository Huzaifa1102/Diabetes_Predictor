import numpy as np

# ==============================
# Example dataset
# ==============================
# Each row represents a mushroom instance.
# Columns: [Heavy, SmellsBad, HasSpots, SmoothSurface, Poisonous]
data = [
    ['No', 'No', 'No', 'No', 'No'],   # A
    ['No', 'No', 'Yes', 'No', 'No'],  # B
    ['Yes', 'Yes', 'No', 'Yes', 'No'],# C
    ['Yes', 'No', 'No', 'Yes', 'Yes'],# D
    ['No', 'Yes', 'Yes', 'No', 'Yes'],# E
    ['No', 'No', 'Yes', 'Yes', 'Yes'],# F
    ['No', 'No', 'No', 'Yes', 'Yes'], # G
    ['Yes', 'Yes', 'No', 'No', 'Yes'] # H
]

# Feature names
attributes = ['Heavy', 'SmellsBad', 'HasSpots', 'SmoothSurface']

# ==============================
# Helper functions
# ==============================

def convert_data(data):
    """
    QUESTION : Add function description
    ANSWER: Converts categorical string data ('Yes', 'No') into numerical binary data (1, 0). This is required because mathematical formulas (like logarithms and arrays) cannot directly compute operations on text.
    """
    mapping = {'Yes': 1, 'No': 0}
    return np.array([[mapping[val] for val in row] for row in data])

def entropy(y):
    """
    QUESTION : Add function description
    ANSWER: Calculates the Shannon entropy of a given set of labels (y).
    QUESTION : What do Entropy measure ?
    ANSWER: Entropy measures the impurity, disorder, or uncertainty in a dataset. A value of 0 means the set is perfectly pure (all one class), while a value of 1 indicates maximum uncertainty (a perfect 50/50 split).
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # ==============================
    # QUESTION : Why do we add + 1e-10 ?
    # ==============================
    return -np.sum(probs * np.log2(probs + 1e-10))  

def information_gain(X, y, feature_index):
    """
    QUESTION : Add function description
    ANSWER: Computes the information gain obtained if the dataset is split on a specific feature.
    QUESTION : How do calculate the information gain ?
    ANSWER: By taking the entropy of the parent dataset (the original uncertainty) and subtracting the weighted average of the entropies of the child subsets created by the split.
    """
    values = np.unique(X[:, feature_index])
    weighted_entropy = 0

    for v in values:
        subset_y = y[X[:, feature_index] == v]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return entropy(y) - weighted_entropy

def build_decision_tree(X, y, attributes):
    """
    QUESTION : Add function description
    """
    # ==============================
    # QUESTION : When do we execute this code ?
    # ANSWER: This is the first base case of our recursion. It executes when all instances in the current node belong to the exact same class (Entropy is 0). It stops splitting and creates a leaf node.
    # ==============================
    if len(np.unique(y)) == 1:
        return y[0]
    # ==============================
    # QUESTION : When do we execute this code ?
    # ANSWER: This is the second base case. It executes when there are no more attributes left to test, but the data is still mixed. It resolves the tie by returning the majority class (the mode) of the current subset.
    # ==============================
    if len(attributes) == 0:
        return np.bincount(y).argmax()
    
    # ==============================
    # QUESTION : What are we computing here ?
    # ANSWER: We are calculating the Information Gain for *every* available feature to evaluate which one splits the data the best (creates the purest sub-nodes).
    # ==============================
    gains = [information_gain(X, y, i) for i in range(X.shape[1])]
    best_feature_index = np.argmax(gains)
    best_feature = attributes[best_feature_index]

    # ==============================
    # QUESTION : What do the variable tree contain ?
    # ANSWER: It initializes a dictionary that represents the current node of the tree, setting the chosen 'best_feature' as the root/key, ready to hold its branches as values.
    # ==============================
    tree = {best_feature: {}}

    # ==============================
    # QUESTION : What are we splitting here ?
    # ANSWER: We are identifying the unique values (e.g., 0 and 1) of the chosen best feature so we can partition the dataset X and labels y into subsets corresponding to each branch.
    # ==============================
    values = np.unique(X[:, best_feature_index])
    for v in values:
        subset_X = X[X[:, best_feature_index] == v]
        subset_y = y[X[:, best_feature_index] == v]
        # ==============================
        # QUESTION : What are we removing (or keeping) here ?
        # ANSWER: We are keeping all attributes EXCEPT the 'best_feature' that was just used for the split. This prevents the tree from testing the exact same attribute again down this specific branch.
        # ==============================
        remaining_attributes = [attr for i, attr in enumerate(attributes) if i != best_feature_index]

        subtree = build_decision_tree(
            subset_X[:, [i for i in range(X.shape[1]) if i != best_feature_index]],
            subset_y,
            remaining_attributes
        )
        tree[best_feature][v] = subtree

    return tree

def predict(tree, instance):
    """
    Add function description
    ANSWER: Recursively traverses the trained decision tree to predict the class label of a new, unseen instance.
    """
    # ==============================
    # QUESTION : When do we execute this code ?
    # ANSWER: When we have reached a leaf node. Leaf nodes contain the final integer prediction (0 or 1), not a dictionary.
    # ==============================
    if not isinstance(tree, dict):
        return tree
    # ==============================
    # QUESTION : What attribute are we getting here ?
    # ANSWER: We are getting the name of the feature (the current node) that we need to evaluate for the test instance.
    # ==============================
    attribute = list(tree.keys())[0]
    attribute_value = instance[attributes.index(attribute)]

    # ==============================
    # QUESTION : What are we doing here ?
    # ANSWER: We are choosing which branch to go down based on the actual value (0 or 1) that the test instance has for this attribute.
    # ==============================
    subtree = tree[attribute].get(attribute_value, None)

    # ==============================
    # QUESTION : When do we execute this ?
    # ANSWER: When the test instance has a feature value that the decision tree never saw during training, meaning no branch exists for it.
    # ==============================
    if subtree is None:
        return None
    
    return predict(subtree, instance)

# ==============================
# QUESTION : What phase is this ?
# ANSWER: The Training (or Learning) phase. This is where the model is built using the historical data.
# ==============================
X = convert_data(data)[:, :-1]  # Features
y = convert_data(data)[:, -1]   # Labels (Poisonous or not)

tree = build_decision_tree(X, y, attributes)
print("Decision Tree built:\n", tree)

# ==============================
# QUESTION : What phase is this ?
# ANSWER: The Testing (or Inference/Prediction) phase. This is where we pass new, unseen data into the trained model to get a prediction.
# ==============================
test_data = [
    ['Yes', 'Yes', 'Yes', 'Yes'],  # U
    ['No', 'Yes', 'No', 'Yes'],    # V
    ['Yes', 'Yes', 'No', 'No'],    # W
]

test_X = convert_data(test_data)

for i, instance in enumerate(test_X):
    pred = predict(tree, instance)
    label = "Poisonous" if pred == 1 else "Not poisonous"
    print(f"Prediction for instance {chr(85 + i)}: {label}")
