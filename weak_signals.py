from scipy.stats import multivariate_normal
import numpy as np

from sklearn.decomposition import SparsePCA

#Generate covariance matrix

Number_of_blocks = 10
blocks_size = np.array([4,4,5,5,6,6,7,7,8,8])
ncol = np.sum(blocks_size)

correlation_within = .9
correlation_between = .1

cov_mat = np.ones((ncol,ncol))*correlation_between

counter = 0
for i in range(Number_of_blocks):
    cov_mat[counter:counter+blocks_size[i],counter:counter+blocks_size[i]]= correlation_within
    counter += blocks_size[i]



np.fill_diagonal(cov_mat,1)

rv=multivariate_normal(np.zeros(ncol),cov_mat)  

n=20         
MW = np.zeros(100)
ML=  np.zeros(100)
from tqdm import tqdm
for j in tqdm(range(100)):
   X=rv.rvs(n)

   SPCA = SparsePCA(n_components=Number_of_blocks)

   Xfit = SPCA.fit(X)

   XPCA = SPCA.transform(X)

   signal = 2*(np.log(ncol)/n)**.5


   c=.8

   beta = np.zeros(ncol)

   true_block = np.array([1,0,0,1,0,0,0,0,0,1])

   counter =0 
   for i in range(Number_of_blocks):
     if true_block[i] == 1:
        beta[counter:counter+blocks_size[i]]=c*signal*np.random.dirichlet(np.ones(blocks_size[i]),1)
     counter+=blocks_size[i]  




   y=np.matmul(X,beta)+np.random.normal(0,1,n) 

   from glmnet import ElasticNet

   lasso1 = ElasticNet()

   lasso1 = lasso1.fit(X,y)


   lasso2 = ElasticNet()

   lasso2 = lasso2.fit(XPCA,y)


   beta_lasso = lasso1.coef_
   beta_weak = np.zeros(ncol)
   for i in range(Number_of_blocks):
     beta_weak = beta_weak + Xfit.components_[i,:]*lasso2.coef_[i]


   Xtest= rv.rvs(100)

   MSE_weak = (np.matmul(Xtest,beta)-np.matmul(Xtest,beta_weak))**2
   MSE_lasso = (np.matmul(Xtest,beta)-np.matmul(Xtest,beta_lasso))**2
   MW[j] =np.average(MSE_weak)
   ML[j] =np.average(MSE_lasso)
