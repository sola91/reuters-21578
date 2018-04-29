import sys
from reuters_reader import ReutersReader
from reuters_preprocessor import ReutersPreprocessor
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def evaluate_model(test_labels, predictions):
    """Calculate the main evaluation criteria for a given model"""
    
    accuracy = accuracy_score(test_labels, predictions)
    hamming = hamming_loss(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    
    print("Global measures")
    print("Accuracy: {:.4f}, Hamming Loss: {:.4f}".format(accuracy, hamming))
    
    
def test_and_evaluate_model(classifier):
    documents = ReutersReader(data_path='data',split='ModApte').get_documents()
    vectorized_train_documents, train_labels, vectorized_test_documents, test_labels = ReutersPreprocessor().pre_process(documents)
    classifier.fit(vectorized_train_documents, train_labels)
    predictions = classifier.predict(vectorized_test_documents)
    evaluate_model(test_labels,predictions)


def main(argv):
   
  if len(argv) > 1:
     if "--linearsvc" == argv[len(argv) - 1]:
         classifier = OneVsRestClassifier(LinearSVC(C=10, loss='hinge'))
         test_and_evaluate_model(classifier)

     elif  "--knn" == argv[len(argv) - 1]:
         classifier = KNeighborsClassifier(p=2, n_neighbors=3,weights="distance")
         test_and_evaluate_model(classifier)

     elif  "--logisticregression" == argv[len(argv) - 1]:
         classifier = OneVsRestClassifier(LogisticRegression(penalty='l1',C=25))
         test_and_evaluate_model(classifier)
      
     elif  "--randomforest" == argv[len(argv) - 1]:
         classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
         test_and_evaluate_model(classifier)

     else:    
         print("Please insert a valid argument")
  else:
     print("Please insert a valid argument")
            
  
if __name__ == "__main__":
    main(sys.argv[0:])