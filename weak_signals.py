from scipy.stats import multivariate_normal
import numpy as np

from sklearn.decomposition import SparsePCA

#Generate covariance matrix
p=1000
n=100
k=5
c=1
Number_of_blocks = 100

blocks_size = np.zeros(Number_of_blocks)
used =0 
for i in range(Number_of_blocks-1):
  theta = (p-used)/(Number_of_blocks-i)
  blocks_size[i] = int(np.random.poisson(theta,1))
  used += blocks_size[i]
blocks_size[Number_of_blocks-1]=p-used
blocks_size=blocks_size.astype(int)


block_idx = np.insert(np.cumsum(blocks_size),0,0)

correlation_within = .9
correlation_between = .1

cov_mat = np.ones((p,p))*correlation_between

counter = 0
for i in range(Number_of_blocks):
    cov_mat[counter:counter+blocks_size[i],counter:counter+blocks_size[i]]= correlation_within
    counter += blocks_size[i]



np.fill_diagonal(cov_mat,1)

rv=multivariate_normal(np.zeros(p),cov_mat)  

SIGNAL = np.zeros(100)         
MW = np.zeros(100)
ML=  np.zeros(100)
TPL = np.zeros(100)
FPL  = np.zeros(100)
TPW  = np.zeros(100)
FPW  = np.zeros(100)

nblocks = np.insert(np.ones(k),np.zeros(Number_of_blocks-k),0)
print(nblocks)


from tqdm import tqdm
for j in tqdm(range(100)):
   true_block = np.random.permutation(nblocks) 	
   X=rv.rvs(n)

   SPCA = SparsePCA(n_components=Number_of_blocks)

   Xfit = SPCA.fit(X)

   XPCA = SPCA.transform(X)

   signal = 2*(np.log(p)/n)**.5


   

   beta = np.zeros(p)

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
   beta_weak = np.zeros(p)
   for i in range(Number_of_blocks):
     beta_weak = beta_weak + Xfit.components_[i,:]*lasso2.coef_[i]


   Xtest= rv.rvs(100)

   MSE_weak = (np.matmul(Xtest,beta)-np.matmul(Xtest,beta_weak))**2
   MSE_lasso = (np.matmul(Xtest,beta)-np.matmul(Xtest,beta_lasso))**2
   SIGNAL[j] = np.average(np.matmul(Xtest,beta)**2)
   MW[j] =np.average(MSE_weak)
   ML[j] =np.average(MSE_lasso)
     
   TPW[j] = np.sum([np.sum(beta_weak[block_idx[i]:block_idx[i+1]]!=0)/blocks_size[i] for i in range(Number_of_blocks) if true_block[i]==1 ] )
   FPW[j] = np.sum([np.sum(beta_weak[block_idx[i]:block_idx[i+1]]!=0)/blocks_size[i] for i in range(Number_of_blocks) if true_block[i]==0 ] )  
 
   TPL[j] = np.sum([np.sum(beta_lasso[block_idx[i]:block_idx[i+1]]!=0)/blocks_size[i] for i in range(Number_of_blocks) if true_block[i]==1 ] )
   FPL[j] = np.sum([np.sum(beta_lasso[block_idx[i]:block_idx[i+1]]!=0)/blocks_size[i] for i in range(Number_of_blocks) if true_block[i]==0 ] )  
 


print(
"SIGNAL = "+str(np.average(SIGNAL))+"\n"+

"MSE LASSO = "+str(np.average(ML))+"\n"+ 

"MSE WEAK = "+str(np.average(MW))+"\n"+  
"POWER LASSO = "+str(np.average(TPL)/np.sum(true_block))+"\n"+
"POWER WEAK = "+str(np.average(TPW)/np.sum(true_block))+"\n"+
"FDR LASSO = "+str(np.average(FPL)/(np.average(FPL)+np.average(TPL)))+"\n"+

"FDR WEAK = "+str(np.average(FPW)/(np.average(FPW)+np.average(TPW)))+"\n")

np.savetxt("sig.csv",SIGNAL,delimiter =";")

np.savetxt("TPL5.csv",TPL,delimiter =";")
np.savetxt("FPL5.csv",FPL,delimiter =";")

np.savetxt("TPW5.csv",TPW,delimiter =";")
np.savetxt("FPW5.csv",FPW,delimiter =";")

np.savetxt("MSEW5.csv",MW,delimiter =";")
np.savetxt("MSEL5.csv",ML,delimiter =";")


