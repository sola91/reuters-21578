import sys
from reuters_reader import ReutersReader
from reuters_preprocessor import ReutersPreprocessor
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
  
def run_grid_search(classifier, parameters):
    documents = ReutersReader(data_path='data',split='ModApte').get_documents()
    vectorized_train_documents, train_labels, _, _ = ReutersPreprocessor().pre_process(documents)
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'f1_micro',
                               cv=10)
    grid_search.fit(vectorized_train_documents, train_labels)
    print ("best f1-weighted score: {0}".format(grid_search.best_score_))
    print ("best parameters: {0}".format(grid_search.best_params_))    



def main(argv):
   
    if len(argv) > 1:
        if "--linearsvc" == argv[len(argv) - 1]:
            parameters = [{'estimator__C' : [1,10,50,100], 'estimator__loss': ['squared_hinge','hinge']}]
            classifier = OneVsRestClassifier(LinearSVC())
            run_grid_search(classifier,parameters)

        elif  "--knn" == argv[len(argv) - 1]:
            parameters = [{'n_neighbors' : [2,3,5], 'weights': ['uniform','distance'], 'p': [1,2]}]
            classifier = KNeighborsClassifier()
            run_grid_search(classifier,parameters)
      
        elif  "--logisticregression" == argv[len(argv) - 1]:
            parameters = [{'estimator__penalty' : ['l1','l2'], 'estimator__C': [5,25,35,50]}]
            classifier = OneVsRestClassifier(LogisticRegression())
            run_grid_search(classifier,parameters)

        elif  "--randomforest" == argv[len(argv) - 1]:
            parameters = [{'n_estimators': np.arange(50,200,50), 'criterion': ['gini','entropy']}]
            classifier = RandomForestClassifier()
            run_grid_search(classifier,parameters)
       
        else:    
            print("Please insert a valid argument")
            
    else:
        print("Please insert a valid argument")
                      
          
if __name__ == "__main__":
    main(sys.argv[0:])