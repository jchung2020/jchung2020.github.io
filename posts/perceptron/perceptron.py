import numpy as np


class Perceptron:
   
    def __init__(self):
        """
        Initialize Perceptron object with attributes:
        - w, weight vector
        - history, tracks score of each weight
        - name
        """
        self.w=[]
        self.history = []
        self.name = "Perceptron"
    
    def fit(self,X,y,max_steps):
        """
        Updates weight w to find hyperplane separating points 
        (represented in X) with labels in y using Perceptron 
        algorithm, terminates in max_steps
        """
        
        n = X.shape[0]
        
        #adds a column of ones to X
        X_ = np.append(X,np.ones((n,1)),axis=1)
        
        #initialize random weight vector, with extra last row for b
        self.w = np.random.rand(X.shape[1]+1)
        
        #updates weight vector w in Perceptron, terminate when
        # accuracy is 1 or max steps are reached
        for t in range(max_steps):
            
            score = self.score(X,y)
            
            self.history.append(score)
            
            #terminate if accuracy is 100%
            #if (score == 1.0):
            if (np.isclose(score,1.0)):
                print("Score is good enough!")
                return
            
            #take a random index i
            i = np.random.randint(n-1)
            
            X_i = X_[i]
            
            y_sign_i = 2*y[i]-1
            
            self.w = self.w + int(np.dot(self.w,X_i)*y_sign_i < 0  )*y_sign_i*X_i
        
        return
            
    def predict(self,X):
        """
        Outputs vector with predictions of labels on data in X
        with current weight vector w
        """
        X_ = np.append(X,np.ones((X.shape[0],1)),axis=1)
        return 1*(X_@self.w >= 0)
    
    def score(self,X, y):
        """
        Returns accuracy (number of correct labels as compared to
        those stored in y) on data in X  with current weight 
        vector w
        """
        return (1*(y==self.predict(X))).sum()*1/X.shape[0]