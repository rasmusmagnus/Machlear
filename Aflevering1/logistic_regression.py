import numpy as np
from h1_util import numerical_grad_check

def logistic(z):
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    """
    logi = np.array([1 / (1 + np.exp(-z[i])) for i in range(len(z))])
    assert logi.shape == z.shape
    return logi


class LogisticRegressionClassifier():

    def __init__(self):
        self.w = None

    def cost_grad(self, X, y, w):
        """
        Compute the average negative log likelihood and gradient under the logistic regression model 
        using data X, targets y, weight vector w 
        
        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features 
           y: np.array shape (n,)  int - Labels 
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar: the average negative log likelihood for logistic regression with data X, y 
           grad: np.arrray shape(d, ) gradient of the average negative log likelihood at w 
        """
        cost = -1/len(y) * sum(y * np.log(logistic(X.dot(w))) + (1 - y) * np.log(1 - logistic(X.dot(w))))
        grad = - 1 / len(y) * X.T.dot(y - logistic(X.dot(w)))
        ### YOUR CODE HERE 5 - 15 lines
        ### END CODE
        assert grad.shape == w.shape
        return cost, grad


    def fit(self, X, y, w=None, lr=0.1, batch_size=16, epochs=10):
        """
        Run mini-batch stochastic Gradient Descent for logistic regression 
        use batch_size data points to compute gradient in each step.
    
        The function np.random.permutation may prove useful for shuffling the data before each epoch
        It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
        Remeber the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

        Args:
           X: np.array shape (n,d) dtype float32 - Features 
           y: np.array shape (n,) dtype int32 - Labels 
           w: np.array shape (d,) dtype float32 - Initial parameter vector
           lr: scalar - learning rate for gradient descent
           batch_size: number of elements to use in minibatch
           epochs: Number of scans through the data

        sets: 
           w: numpy array shape (d,) learned weight vector w
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        b = batch_size
        if w is None: w = np.zeros(X.shape[1])
        history = []        
        ### YOUR CODE HERE 14 - 20 lines
        for i in range(epochs):
            p = np.random.permutation(range(len(y)))
            for j in range(int( len(y) / batch_size + 0.5)):
                w = w - lr * self.cost_grad(np.array([X[i] for i in p[b*j : b*(j + 1)]]), np.array([y[i] for i in p[b*j : b * (j + 1)]]), w)[1]
            history = history + [self.cost_grad(X, y, w)[0]]
                        
        ### END CODE
        self.w = w
        self.history = history


    def predict(self, X):
        """ Classify each data element in X

        Args:
            X: np.array shape (n,d) dtype float - Features 
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on X (0, 1)

        """
        #f rounds the paramteter to 0 or 1.
        f = lambda x : [0 if x[i] < 0.5 else 1 for i in range(len(X))]
        out = np.array(f(logistic(X.dot(self.w))))
        ### YOUR CODE HERE 1 - 4 lines
        ### END CODE
        return out
    
    def score(self, X, y):
        """ Compute model accuracy  on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features 
            y: np.array shape (n,) dtype int - Labels 

        Returns: 
           s: float, number of correct prediction divivded by n.

        """
        s = 1 - 1 / len(y) * sum(abs(y - self.predict(X)))
        ### YOUR CODE HERE 1 - 4 lines
        ### END CODE
        return s
        

    
def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([0, 0, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])    
    w = np.array([0.0, 0.0])
    y = np.array([0, 0, 1]).astype('int64')
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    numerical_grad_check(f, w)
    print('Test Success')


    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    
    
