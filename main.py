import sys
from reuters_reader import ReutersReader
from reuters_preprocessor import ReutersPreprocessor
 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def SVC_classifier(vec_train_documents, train_labels, parameters):
    classifier = OneVsRestClassifier(LinearSVC(C=1))
    classifier.fit(vec_train_documents, train_labels)
    return classifier

def evaluate(test_labels, predictions):
    accuracy = accuracy_score(test_labels, predictions)
    hamming = hamming_loss(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')

    print("Weighted-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    
    print("Global measures")
    print("Accuracy: {:.4f}, Hamming Loss: {:.4f}".format(accuracy, hamming))
    
    
def run_grid_search(name, classifier, parameters,vec_train_documents, train_labels):
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'f1_weighted',
                               cv=5)
    grid_search.fit(vec_train_documents, train_labels)
    return grid_search.best_score_ , grid_search.best_params_



def main(argv):
   
    documents = ReutersReader('data').get_documents()
    vec_train_documents, vec_test_documents, train_labels, test_labels = ReutersPreprocessor().pre_process(documents)

    if len(argv) > 1:
        if "--linearsvc" == argv[len(argv) - 1]:
            parameters = [{'estimator__C' : [1,10,50,100], 'estimator__loss': ['squared_hinge','hinge']}]
            classifier = OneVsRestClassifier(LinearSVC())
            best_score, best_params = run_grid_search('LinearSVC', classifier,
                    parameters, vec_train_documents, train_labels)
            print ("best f1-weighted score: {0}".format(best_score))
            print ("best parameters: {0}".format(best_params))
        if  "--knn" == argv[len(argv) - 1]:
            parameters = [{'n_neighbors' : [5,10,15,25,50,75], 'weights': ['uniform','distance']}]
            classifier = KNeighborsClassifier()
            best_score, best_params = run_grid_search('KNeighborsClassifier', classifier,
                    parameters, vec_train_documents, train_labels)
            print ("best f1-weighted score: {0}".format(best_score))
            print ("best parameters: {0}".format(best_params))
           
            
            classifier = KNeighborsClassifier(n_neighbors=75,weights="uniform")
            classifier.fit(vec_train_documents, train_labels)
            predictions = classifier.predict(vec_test_documents)
            evaluate(test_labels,predictions)
            
            
            
if __name__ == "__main__":
    main(sys.argv[0:])