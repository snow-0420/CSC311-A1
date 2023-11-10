from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

#imports for plotting
from sklearn import tree
from matplotlib import pyplot as plt


"""a random seed used to generate same split of data in different run"""
random_seed = 35

def load_data():
    """load the data and process it using vectorizer, as well as split
    the data into training, validation and test sets"""

    text_fake = open("clean_fake.txt", 'r')
    text_real = open("clean_real.txt", 'r')

    """list of strings, where each string is a title"""
    data_fake = text_fake.readlines()
    data_real = text_real.readlines()

    """0 to indicate fake, 1 to indicate real as class label"""
    class_label = [0] * len(data_fake) + [1] * len(data_real)

    """combined data"""
    data = data_fake + data_real

    """initiate vectorizer"""
    vect = CountVectorizer()

    """data in terms of matrix (a row represent a title, 
    a column represent a single word as a feature,
    and value of an entry represent the occurrence count of the word
    in that title)"""
    data_matrix = vect.fit_transform(data)

    """use for plotting and calculating information gain"""
    feature_names = vect.get_feature_names_out()

    """Separate the data into 70% training, 15% validation and 15% test
    (as well as the class_label list in the same way"""
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    data_train, data_val_test, label_train, label_val_test = \
        train_test_split(data_matrix, class_label, train_size=train_ratio, random_state=random_seed)
    data_val, data_test, label_val, label_test = \
        train_test_split(data_val_test, label_val_test, train_size=val_ratio/(val_ratio+test_ratio), random_state=random_seed)

    """use for calculating information gain"""
    data_train_text = vect.inverse_transform(data_train)

    """return the data processed by vectorizer, and all the separated datasets"""
    return data_matrix, data_train, data_val, data_test, label_train, label_val, label_test, feature_names, data_train_text


def train_test_val(max_depth, criterion, data_train, data_val, label_train, label_val, feature_names):
    """train the data with given parameters and test on validation data
    input:
    max_depth: int representing the max depth of the tree
    criterion: string parameter for training the classifier
    data_train, data_val, label_train, label_val: training and validation data
        with their corresponding class labels
    feature_names: used only for plotting
    return: accuracy"""

    """initiate the decision tree classifier"""
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_seed)

    """train the decision tree using the training data"""
    clf.fit(data_train, label_train)

    """test on validation data"""
    num_total = len(label_val)
    num_correct = 0
    predict = clf.predict(data_val)
    for i in range(len(predict)):
        if predict[i] == label_val[i]:
            num_correct += 1

    accuracy = num_correct / num_total

    """for plotting"""
    #tree.plot_tree(clf, max_depth=2, class_names=["fake", "true"], filled=True, rounded=True, feature_names=feature_names)
    #plt.savefig('pic.pdf')

    return accuracy


def select_model():
    """use 15 different configuration (3 different criteria,
    5 different max depth) to evaluate accuracy for different configuration
    prints accuracy and corresponding configuration to stdout"""

    """load data"""
    data_matrix, data_train, data_val, data_test, label_train, label_val, label_test, feature_names, data_train_text = load_data()

    """setup different configurations"""
    criteria = ["gini", "entropy", "log_loss"]
    depths = [5, 15, 30, 50, 100]

    """best configuration"""
    #criteria = ["gini"]
    #depths = [100]

    """for plotting"""
    #accuracies = []

    """evaluation!"""
    for criterion in criteria:
        for max_depth in depths:
            accuracy = train_test_val(max_depth, criterion, data_train, data_val, label_train, label_val, feature_names)
            print("With {1} criteria and max depth of {2}: {0} \n".format(accuracy, criterion, max_depth))

            """for plotting"""
            #accuracies.append(accuracy)

    """for plotting"""
    #fig, ax = plt.subplots()
    #ax.set_title("Accuracy by Max Depth on Different Criterion")
    #ax.plot([1, 2, 3, 4, 5], accuracies[0:5], label="gini")
    #ax.plot([1, 2, 3, 4, 5], accuracies[5:10], label="entropy")
    #ax.plot([1, 2, 3, 4, 5], accuracies[10:15], label="log_loss")
    #ax.set_xlabel("Max depth")
    #ax.set_ylabel("Accuracy")
    #ax.set_xticks([1, 2, 3, 4, 5], depths)
    #ax.legend()
    #plt.show()


def compute_information_gain(data_train, label_train, feature_names):
    """uses different keywords (I choose some other words shown in the graph
    for q2c, as well as some random words that's known to be in some titles)
    as splits to compute information gain

    inputs:
    data_train, label_train: training set and corresponding labels
    (notice the training set is represented as list of str in this
    specific case, allowing manual calculation of information gain
    without using decision tree classifier API)
    feature_names: a list of feature names (i.e. all different keywords)

    return: none, only print the information gain calculated
    on the chosen splits"""

    """Some basic data (for simplicity, I'll manually write those values)"""
    total = 2286 #the number of titles in the training set
    count_fake = 910 #the number of fake titles in the training set
    count_real = 1376 #the number of real titles in the training set

    """the respective discrete distribution"""
    prob = [count_fake/total, count_real/total]

    """entropy of the unsplitted training set"""
    entropy_y = entropy(prob, base=2)

    """list of keywords we are going to use"""
    keywords = ["the", "hillary", "trump"]

    """randomly get two other keywords"""
    np.random.seed(random_seed)
    random_index = np.random.randint(len(feature_names), size=2)
    for index in random_index:
        keywords.append(feature_names[index])

    """calculate the entropy and information gain for each split"""
    for split in keywords:
        #initiating counters
        left_fake, left_real, right_fake, right_real = 0, 0, 0, 0

        #do the split and count numbers of each category
        for index in range(len(data_train)):
            if split in data_train[index]: #if title contains the keyword
                if label_train[index] == 0: #if title is fake
                    right_fake += 1
                else: #if title is real
                    right_real += 1
            else: #if title does not contain the keyword
                if label_train[index] == 0: #if title is fake
                    left_fake += 1
                else: #if title is real
                    left_real += 1

        count_left = left_fake + left_real
        count_right = right_fake + right_real

        #probability of Y given contains the word
        prob_left = [left_fake/count_left, left_real/count_left]
        #probability of Y given doesn't contain the word
        prob_right = [right_fake/count_right, right_real/count_right]

        #entropy of Y given contains the word
        entropy_left = entropy(prob_left, base=2)
        #entropy of Y given doesn't contain the word
        entropy_right = entropy(prob_right, base=2)

        #entropy of Y given X
        entropy_y_given_x = count_left/total * entropy_left + count_right/total * entropy_right

        #information gain of Y given X
        ig_y_given_x = entropy_y - entropy_y_given_x

        #print the result
        print("The information gain given the split on '{0}': {1}".format(split, ig_y_given_x))
