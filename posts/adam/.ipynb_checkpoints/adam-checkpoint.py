import numpy as np

class Adam:
   
    def __init__(self):
        """
        Initialize LogisticRegression object with attributes:
        - w, weight vector
        - history, tracks score of each weight
        - name
        """
        self.w=[] #includes bias term b
        self.loss_history = []
        self.history = []
        self.name = "Logistic Regression"
        
    def sigmoid(self,z):
        """
        Returns sigmoid function, to be used in the logistic loss
        """
        return 1/(1+np.exp(-z))
    
    def logistic_loss(self,y_pred,y):
        """
        Convex loss function for logistic regression
        """
        return -y*np.log(self.sigmoid(y_pred)) - (1-y)*np.log(1-self.sigmoid(y_pred))
    
    def gradient(self,y_pred,y,X):
        """
        Returns gradient of the logistic loss function
        - This is from the lecture notes, the X.T is to ensure the correct form
        - Note this X should be X_ with the column of 1s
        """
        
        gradient = np.zeros(X.shape[1])
        for i in range(len(y)):
            gradient += (self.sigmoid(y_pred[i]) - y[i])*X[i]
            
        gradient = gradient/(X.shape[0])
        
        if not np.allclose(X.T@(self.sigmoid(y_pred) - y)*1/(X.shape[0]), gradient):
            print("Gradients not close!")
        
        return gradient
        
        #return X.T@(self.sigmoid(y_pred) - y)*1/(X.shape[0])
    
    def fit(self,X,y,alpha,max_epochs):
        """
        Uses gradient descent to determine weights w with minimum loss
        - X is the feature matrix
        - y is the target vector
        - alpha is the step size for gradient descent
        - max_epochs is the maximum number of steps
        """
        
        n = X.shape[0]
        
        #adds a column of ones to X
        X_ = np.append(X,np.ones((n,1)),axis=1)
        
        #initialize random weight vector, with extra last row for b
        self.w = np.random.rand(X.shape[1]+1)
        
        #updates weight vector w in LogisticRegression, terminate when
        # gradient is 0 or max steps are reached
        for t in range(max_epochs):
            
            score = self.score(X_,y) #SHOULD I REALLY DO THIS BEFORE THE w UPDATE?
            self.history.append(score)
            
            loss = self.loss(X_,y)
            self.loss_history.append(loss)
            
            y_pred = self.predict(X_)
            grad = self.gradient(y_pred,y,X_)
            
            #stop if gradient is zero
            if (np.allclose(grad,np.zeros(len(grad)))):
            #if (np.isclose(grad.sum(),0.0)):
                return
            
            self.w -= alpha*grad
        
        return
    
    def fit_stochastic(self,X,y,m_epochs,momentum,batch_size,alpha):
        """
        Uses stochastic gradient descent to determine weights w with minimum loss
        - X is the feature matrix
        - y is the target vector
        - alpha is the step size for gradient descent
        - m_epochs is the maximum number of steps
        - momentum optionally uses momentum
        - batch_size determines the size of the batches
        """
        
        n = X.shape[0]
        
        #adds a column of ones to X
        X_ = np.append(X,np.ones((n,1)),axis=1)
        
        #initialize random weight vector, with extra last row for b
        self.w = np.random.rand(X.shape[1]+1)
        
        #w_prev is used for momentum
        w_prev = self.w
        
        #only set momentum parameter if specified
        if (momentum):
            beta = 0.8 
        else:
            beta = 0.0
        
        #updates weight vector w in LogisticRegression, terminate when
        # accuracy is 1 or max steps are reached
        for t in range(m_epochs):
            
            order = np.arange(n)
            np.random.shuffle(order)
            
            for batch in np.array_split(order, n//batch_size+1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                
                y_pred = self.predict(x_batch)
                grad = self.gradient(y_pred,y_batch,x_batch)
                
                w_copy = self.w
                self.w = self.w - alpha*grad + beta*(self.w - w_prev)
                w_prev = w_copy
                
                #again, return if gradient is 0
                #if (np.allclose(grad,np.zeros(len(grad)))):
                if (np.allclose(self.w,w_prev)):
                    print("Iterations done at ",t)
                    print(w_prev)
                    print(self.w)
                    return
                
            score = self.score(X_,y)
            self.history.append(score)
            
            loss = self.loss(X_,y)
            self.loss_history.append(loss)

        return  
    
    def fit_adam(self,X,y,batch_size,alpha,beta_1,beta_2,w_0 = None,m_epochs = None):
        """
        Uses stochastic gradient descent to determine weights w with minimum loss
        - X is the feature matrix
        - y is the target vector
        - alpha is the step size for gradient descent
        - m_epochs is the maximum number of steps
        - momentum optionally uses momentum
        - batch_size determines the size of the batches
        """
        
        n = X.shape[0]
        
        #adds a column of ones to X
        X_ = np.append(X,np.ones((n,1)),axis=1)
        
        if w_0 == None:
            #initialize random weight vector, with extra last row for b
            #self.w = np.random.rand(X.shape[1]+1)
            w_0 = np.random.rand(X.shape[1]+1)
        
        self.w = w_0
        
        #w_prev is used for momentum
        w_prev = np.ones(X.shape[1]+1)
        
        m_t = np.zeros(X.shape[1]+1)
        v_t = np.zeros(X.shape[1]+1)
        t = 0
        
        g_t = np.ones(X.shape[1]+1)
        
        epsilon = 10.0**(-8) #as recommended by the paper
        
        #m_prev = m_t
        #v_prev = v_t
        
        #updates weight vector w in LogisticRegression, terminate when
        # accuracy is 1 or max steps are reached
        
        while not np.allclose(self.w,w_prev):
        #while not np.allclose((self.w-w_prev),np.zeros(X.shape[1]+1)): #THIS WILL LOOP FOR LONG TIME
        #while not np.allclose(g_t,np.zeros(X.shape[1]+1)) or t!=0:
            
            t += 1
            
            w_prev = self.w #IMPORTANT, this is so while loop stops
            
            order = np.arange(n)
            np.random.shuffle(order)
            
            for batch in np.array_split(order, n//batch_size+1):
                x_batch = X_[batch,:] #notice X here is not padded
                y_batch = y[batch]
                
                y_pred = self.predict(x_batch)
                
                g_t = self.gradient(y_pred,y_batch,x_batch)
                m_t = beta_1*m_t + (1 - beta_1)*g_t
                v_t = beta_2*v_t + (1 - beta_2)*(g_t**2)
                m_t_hat = m_t/(1-beta_1**(t))
                v_t_hat = v_t/(1-beta_2**(t))
                
                self.w = self.w - alpha*m_t_hat/(np.sqrt(v_t_hat) + epsilon)
                
                
                """
                w_copy = self.w
                self.w = self.w - alpha*grad + beta*(self.w - w_prev)
                w_prev = w_copy
                
                #again, return if gradient is 0
                if (np.allclose(grad,np.zeros(len(grad)))):
                    return
                """
            
            #"""
            if (t % 1000 == 0):
                print(t," weights: ",self.w)
                print(t, " prev weights: ",w_prev)
                print(t, " gradient: ",g_t)
            #"""  
           
            """
            score = self.score(X,y)
            self.score_history.append(score)
            """
               
            #"""
            score = self.score(X_,y)
            self.history.append(score)
            
            loss = self.loss(X_,y)
            self.loss_history.append(loss)
            #"""
            
            if (t == m_epochs):
                return
        
        print("t: ",t)
        print("prev weights: ",w_prev)
        print("diff weights: ",w_prev-self.w)

        return    
        
    def predict(self,X):
        """
        Outputs vector with predictions of labels on data in X
        with current weight vector w
        - Here X is assumed to be X_, with a column on 1s
        """
        return X@self.w
    
    def score(self,X, y):
        """
        Returns accuracy (number of correct labels as compared to
        those stored in y) on data in X  with current weight 
        vector w
        - Here X is assumed to be X_, with a column on 1s
        """
        #Our condition here is that a positive sign for our prediction is 1, rest is 0
        return (1*(y == 1*(self.predict(X)>0))).mean()
    
    def loss(self,X,y):
        """
        Calculates loss (in this case logistic regression loss)
        - Here X is assumed to be X_, with a column on 1s
        """
        y_hat = self.predict(X)
        return self.logistic_loss(y_hat,y).mean() #remember that y_hat and y can be vectors
                
    
    def pad(self,X):
        """
        Adds a column of 1s to the the feature matrix X
        """
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)