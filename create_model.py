# cgs
def cgs(A):
  """
    Q,R = cgs(A)
    Apply classical Gram-Schmidt to mxn rectangular/square matrix. 

    Parameters
    -------
    A: mxn rectangular/square matrix   

    Returns
    -------
    Q: mxn square matrix
    R: nxn upper triangular matrix

  """
  # ADD YOUR CODES
  m=A.shape[0] # get the number of rows of A
  n=A.shape[1] # get the number of columns of A

  R=np.zeros((n,n)) # create a zero matrix of nxn
  Q=np.ones((m,n)) # copy A (deep copy)
  for k in range(n):
    w=A[::,k]
    for j in range(0,k-1):
      R[j][k]=np.matmul(np.transpose(Q[::,j]),w)
    #for j in range(0,k-1):
      w=w-R[j][k]*Q[::,j]
    R[k][k]=np.linalg.norm(w,2)
    Q[::,k]=w/R[k][k]

  return Q,R


  
  # Implement BACK SUBS
def backsubs(U, b):

  """
  x = backsubs(U, b)
  Apply back substitution for the square upper triangular system Ux=b. 

  Parameters
  -------
    U: nxn square upper triangular array
    b: n array
    

  Returns
  -------
    x: n array
  """

  n= U.shape[1]
  x= np.zeros((n,))
  b_copy= np.copy(b)

  if U[n-1,n-1]==0.0:
    if b[n-1] != 0.0:
      print("System has no solution.")
  
  else:
    x[n-1]= b_copy[n-1]/U[n-1,n-1]
  for i in range(n-2,-1,-1):
    if U[i,i]==0.0:
      if b[i]!= 0.0:
        print("System has no solution.")
    else:
      for j in range(i,n):
        b_copy[i] -=U[i,j]*x[j]
      x[i]= b_copy[i]/U[i,i]
  return x

  # Add ones
def add_ones(X):
  

  # ADD YOUR CODES
  m,n=X.shape
  #return np.concatenate((np.ones((m,1)),X),axis=1)
  return np.hstack((np.ones((m,1)),X))


## Add ones to X
X=add_ones(X)

def split_data(X,Y, train_size):
  # ADD YOUR CODES
  # shuffle the data before splitting it

  Y=Y.reshape(-1,1)
  m,n=X.shape
  train_size=int(m*train_size)
  data=np.concatenate((X,Y),axis=1)

  np.random.shuffle(data)

  #data_training=data[:train_size]
  #data_test=data[train_size:]

  X_train=data[:train_size,:n]
  X_test=data[train_size:,:n]

  y_train=data[:train_size][::,-1]
  y_test=data[train_size:][::,-1]

  return X_train,X_test,y_train,y_test
 # pass


 # Split (X,y) into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test=split_data(X,y, train_size=0.8)

def mse(y, y_pred):
  
    # ADD YOUR CODES
    return np.mean((y-y_pred)*(y-y_pred))
    


def normalEquation(X,y):
   
 if np.linalg.det(np.matmul(np.transpose(X),X))!=0:
    return np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y))




class LinearRegression:

  def __init__(self, arg):
      # ADD YOUR CODES
      self.arg=arg
      
  def fit(self,x,y):
      # ADD YOUR CODES
      #x=add_ones(x)
      self.x=x
      self.y=y
      if self.arg=="normalEquation":
        theta=normalEquation(self.x,self.y)
        self.theta=theta

      if self.arg=="qrFactorization":
        theta=backsubs(cgs(self.x)[1], np.matmul(np.transpose(cgs(self.x)[0]),self.y))
        self.theta=theta
    

  def predict(self,x):
    
      #ADD YOUR CODES
      y_pred=np.dot(x,self.theta)
      self.y_pred=y_pred
      #return y_pred