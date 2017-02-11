rm(list=ls())

# Install and require packages
PackageList =c('caret','mlbench','Hmisc','randomForest','data.table')
NewPackages=PackageList[!(PackageList %in%
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE)

# DGP: 5 true + 45 pure noise variables
n <- 100
p <- 40
sigma <- 1
set.seed(10)
sim <- mlbench.friedman1(n, sd = sigma)
colnames(sim$x) <- c(paste("real", 1:5, sep = ""),
                     paste("bogus", 1:5, sep = ""))
bogus <- matrix(rnorm(n * p), nrow = n)
colnames(bogus) <- paste("bogus", 5+(1:ncol(bogus)), sep = "")
x <- cbind(sim$x, bogus)
y <- sim$y

# predictors centered and scaled
normalize <- function(x){
        (x-mean(x))/sd(x)
}
x <- apply(x, 2, normalize)
x <- as.data.table(x)
dt <- cbind(x,y)

# Number of variables to try
subsets <- c(1:5, 10, 15, 20, 25)

# Resampling routine
n_cv=5
n_fold=10

RMSE=array(0, dim=c(n_cv, n_fold, length(subsets)+1))
varImp_mat=array(0, dim=c(n_cv, n_fold, dim(x)[2]))

for (i in 1:n_cv){
        # shuffle the rows
        ii=sample(1:n, n)
        dt_sf=dt[ii,]
        fs=round(n/n_fold)
        for (j in 1:n_fold){
                bot=(j-1)*fs+1; top=ifelse(j==n_fold,n,j*fs);ii=bot:top
                train=dt[-ii,]
                test=dt[ii,] # hold-out set
                
                # Train rf model using all predictors
                rf=randomForest(y~., 
                                data=train,
                                ntree=500,
                                mtry=sqrt(p+10),
                                importance=TRUE)
                
                # Predict the hold-out sample
                yhat_all=predict(rf, newdata=test)
                RMSE[i,j,length(subsets)+1]=sqrt(mean((test$y-yhat_all)^2))
                
                # Calculate variable importance/rankings
                varImp=importance(rf) # a matrix with two types of measures
                varImp_mat[i,j,]=varImp[,2]
                
                # Sort
                sort_idx=order(-varImp[,2])
                varImp_sorted=varImp[sort_idx,]
                varNamesSorted <- row.names(varImp_sorted) # return the var names in order
                
                for (k in length(subsets):1){
                        
                        # Number of vars
                        s=subsets[k]
                        cat("fold",j,".rep",i,"size: ",s,'\n')
                        
                        # Keep the s most important vars
                        pred_vars=varNamesSorted[1:s]
                        
                        # Train the model
                        #formula(paste("y ~ ", paste(pred_vars, collapse = "+")))
                        rf_sub=randomForest(x=train[,pred_vars, with=FALSE],
                                            y=train$y,
                                            ntree=500,
                                            mtry=sqrt(s),
                                            importance=TRUE)
                        
                        # Predict on hold-out sample
                        yhat=predict(rf_sub, newdata=test)
                        RMSE[i,j,k]=sqrt(mean((test$y-yhat)^2))
                }
        }
}

# Calculate performance profile over S_i
(RMSE_mat=cbind(c(subsets,50), apply(RMSE, 1, colMeans)))
colnames(RMSE_mat) <- c("varNum", paste0("rep", 1:n_cv))
RMSE_mat

ave_RMSE=apply(RMSE_mat[,2:6], 1, mean)

plot(c(subsets,50), ave_RMSE, type="l", col="blue", lwd=2, cex.lab=1,
     xlab="Number of variables", ylab="RMSE (repeated cross-validation)")

# Determine the appropriate number of predictors
varNum_opt=subsets[which.min(ave_RMSE)] 
cat("Optimal number of predictors: ", varNum_opt)

# Predictors to include in the final model
varImp_mat=apply(varImp_mat, 1, colMeans)
ave_varImp=apply(varImp_mat, 1, mean)

# Sort
varImp_sorted=varImp[order(-ave_varImp),]
varNamesSorted <- row.names(varImp_sorted) # return the var names in order
pred_vars=varNamesSorted[1:varNum_opt]

# Fit the final model based on the optimal S_i using the original training set
rf_opt=randomForest(x=dt[,pred_vars, with=FALSE],
                    y=dt$y,
                    ntree=500,
                    importance=TRUE)
print(rf_opt)
varImpPlot(rf_opt)