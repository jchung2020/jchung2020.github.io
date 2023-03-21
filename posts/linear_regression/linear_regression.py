import numpy as np

class LinearRegression:
    
    def __init__(self):
        """
        Initialize LinearRegression Object
        Atrributes:
        - name
        - weight vector w
        - score history
        """
        self.name = "LinearRegression"
        self.w = np.array([])
        self.score_history = []
        
    def fit_analytic(self,X,y):
        """
        Takes in feature matrix X and target vector y, finds weight vector
        w using the analytic method (eq. 3 in Least-Squares Linear Regression
        Notes)
        """
        
        #adds a column of ones to X
        X_ = self.pad(X)
        
        P = X_.T@X_
        q = X_.T@y
        
        self.w = np.linalg.inv(P)@q
        
        
        
    def fit_gradient(self,X,y,max_iter,alpha):
        """
        Takes in feature matrix X and target vector y, max iterations,
        learning rate alpha, and finds weight vector w using gradient 
        descent 
        """
        
        #adds a column of ones to X
        X_ = self.pad(X)
        
        #initialize random weight vector, with extra last row for b
        self.w = np.random.rand(X.shape[1]+1)
        
        P = X_.T@X_
        q = X_.T@y
        
        for i in range(max_iter):
            
            gradient = 2.0*(P@self.w-q)
            
            self.w -= alpha*gradient
            score = self.score(X,y)
            self.score_history.append(score)
            
            #exit loop if gradient is close to zero
            if (np.allclose(gradient,np.zeros(len(gradient)))):
                print("All good! Iteration ",i)
                break
        
        
    def predict(self,X):
        """
        Returns prediction vector of target,
        for Linear Regression, this is just
        X times the weight vector
        """
        return self.pad(X)@self.w #note that X is padded with a 1s column
        
    def score(self,X,y):
        """
        Computes score for Linear Regression, actually the
        coefficient of determination
        """
        y_pred = self.predict(X)
        y_mean = y.mean()
        return 1 - ((y_pred-y)**2).sum()/((y_mean*np.ones(y.size)-y)**2).sum()
    
    def pad(self,X):
        """
        Adds a column of 1s to the the feature matrix X
        """
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

        