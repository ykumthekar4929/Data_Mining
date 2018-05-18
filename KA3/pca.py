import numpy as np

# A= [8,0,10,10,2],[-20,-1,-19,-20,0]



np.random.seed(12345) ## 12345 this is the seed


""" 5 data instance ,3-dim feature """
A = np.random.random((5,3))


a= np.array(A)
mean_row=np.array(a.mean(1))
a=a-(mean_row.reshape(-1,1))
mean = np.mean(a)
covar=np.cov(a)
covar=(covar*(a.shape[1]-1))
covar=covar/(a.shape[1])

print("Covariance matrix\n")
print(covar)

print("\n Eigen values, eigen vectors")
print(np.linalg.eig(covar))
