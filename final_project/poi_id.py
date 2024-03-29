#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'other', 'exercised_stock_options', 
	'restricted_stock', 'from_poi_to_this_person', 'from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
	data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

features = StandardScaler().fit_transform(features)

pca = PCA(n_components = 3)
pca.fit(features)
print(pca.explained_variance_ratio_)
print()
first_pc = pca.components_[0]
second_pc = pca.components_[1]
transformed_data = pca.transform(features)
for ii, jj in zip(transformed_data, features):
	plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color = 'r')
	plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1], color = 'c')
	plt.scatter(jj[0], jj[1], color = 'b')
plt.show()

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
from sklearn.ensemble import RandomForestClassifier as RFC
clf = RFC(n_estimators = 100, max_features = 'auto', random_state = 0)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
#print(accuracy_score(labels_test, predictions))

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0

# cv = StratifiedShuffleSplit(n_splits = 50, test_size = 0.3, random_state = 42)
# for train_idx, test_idx in cv.split(features, labels): 
	# features_train = []
	# features_test  = []
	# labels_train   = []
	# labels_test	   = []
	# for ii in train_idx:
		# features_train.append( features[ii] )
		# labels_train.append( labels[ii] )
	# for jj in test_idx:
		# features_test.append( features[jj] )
		# labels_test.append( labels[jj] )
		
	# clf.fit(features_train, labels_train)
	# predictions = clf.predict(features_test)
		
for prediction, truth in zip(predictions, labels_test):
	if prediction == 0 and truth == 0:
		true_negatives += 1
	elif prediction == 0 and truth == 1:
		false_negatives += 1
	elif prediction == 1 and truth == 0:
		false_positives += 1
	elif prediction == 1 and truth == 1:
		true_positives += 1
	else:
		print("Warning: Found a predicted label not == 0 or 1.")
		print("All predictions should take value 0 or 1.")
		print("Evaluating performance for processed predictions:")
		break

try:
	total_predictions = true_negatives + false_negatives + false_positives + true_positives
	accuracy = 1.0*(true_positives + true_negatives)/total_predictions
	precision = 1.0*true_positives/(true_positives+false_positives)
	recall = 1.0*true_positives/(true_positives+false_negatives)
	f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
	f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
	print()
	print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
	print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
	print("")
except:
	print("Got a divide by zero when trying out:", clf)
	print("Precision or recall may be undefined due to a lack of true positive predicitons.")


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)