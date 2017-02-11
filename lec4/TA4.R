######################################################################
# Download and pre-process data
download.file(
  'https://github.com/ChicagoBoothML/MLClassData/raw/master/GiveMeSomeCredit/CreditScoring.csv',
  'CreditScoring.csv')

trainDf = read.csv("CreditScoring.csv")

##remove X (is 1:n)
trainDf = trainDf[,-1]

##add y as factor
trainDf$y = as.factor(trainDf$SeriousDlqin2yrs)
trainDf = trainDf[,-1] # get rid of old y = SeriousDlqin2yrs

##get rid of NumberOfDependents, don't want to deal with NA's 
trainDf=trainDf[,-10]
##get rid of MonthlyIncome, don't want to deal with NA's 
trainDf=trainDf[,-5]

##split train in train, test
## in class demo we only use a quarter of the data
n=nrow(trainDf) 
set.seed(99)
ii = sample(1:n,n)
ntest = floor(n/2)
testDf = trainDf[ii[1:ntest],]
trainDf = trainDf[ii[(ntest+1):n],]


######################################################################
# Summary the data
table(trainDf$y)
# 4979 customers default

plot(age~y,trainDf,col=c("red","blue"),cex.lab=1.4)


######################################################################
# Preparing for Modeling
library(ROCR)
library(tree)
library(randomForest)
library(gbm)

# define loss function
# deviance loss function
# y should be 0/1
# phat are probabilities obtain by our algorithm 
# wht shrinks probs in phat towards .5 --- this helps avoid numerical problems don't use log(0)!
lossf = function(y,phat,wht=0.0000001) {
  if(is.factor(y)) y = as.numeric(y)-1
  phat = (1-wht)*phat + wht*.5
  py = ifelse(y==1, phat, 1-phat)
  return(-2*sum(log(py)))
}

# deviance loss function
# y should be 0/1
# phat are probabilities obtain by our algorithm 
# thr is the cut off value - everything above thr is classified as 1
getConfusionMatrix = function(y,phat,thr=0.5) {
  # some models predict probabilities that the data belongs to class 1,
  # This function convert probability to 0 - 1 labels
  if(is.factor(y)) y = as.numeric(y)-1
  yhat = ifelse(phat > thr, 1, 0)
  tb = table(predictions = yhat, 
             actual = y)  
  rownames(tb) = c("predict_0", "predict_1")
  return(tb)
}



# deviance loss function
# y should be 0/1
# phat are probabilities obtain by our algorithm 
# thr is the cut off value - everything above thr is classified as 1
lossMR = function(y,phat,thr=0.5) {
  if(is.factor(y)) y = as.numeric(y)-1
  yhat = ifelse(phat > thr, 1, 0)
  return(1 - mean(yhat == y))
}


# initialize a place to store results of multiple models.
phatL = list()



######################################################################
# Logistic Regression
lgfit = glm(y~., trainDf, family=binomial)
print(summary(lgfit))
phat = predict(lgfit, testDf, type="response")
phatL$logit = matrix(phat,ncol=1) 




######################################################################
# Random Forest
set.seed(99)
# We want to fit several different random forest models with different tuning parameters setting
# There are two key parameters
# mtry : number of variables randomly sampled as candidates at each splits
# ntree : number of trees in the random forest
##settings for randomForest
p=ncol(trainDf)-1 # number of variables 
mtryv = c(p, sqrt(p)) 
ntreev = c(500,1000) # number of trees
(setrf = expand.grid(mtryv,ntreev))  # this contains all settings to try
colnames(setrf)=c("mtry","ntree")
setrf # a matrix, each row is a parameter setting for random forest

# initialize a place to store fitting results
phatL$rf = matrix(0.0,nrow(testDf),nrow(setrf))  # we will store results here

###fit rf
for(i in 1:nrow(setrf)) { # loop over all parameter settings
  #fit and predict
  frf = randomForest(y~., data=trainDf, 
                     mtry=setrf[i,1], 
                     ntree=setrf[i,2], 
                     nodesize=10)
  phat = predict(frf, newdata=testDf, type="prob")[,2]
  phatL$rf[,i]=phat
}





######################################################################
# Boosting
# set variables 
# There are three key parameters in boosting
# tdepth : 	The maximum depth of variable interactions
# ntree : number of trees 
# shrink : a shrinkage parameter applied to each tree in the expansion
##settings for boosting
idv = c(2,4)
ntv = c(1000,5000)
shv = c(.1,.01)
(setboost = expand.grid(idv,ntv,shv))
colnames(setboost) = c("tdepth","ntree","shrink")
phatL$boost = matrix(0.0,nrow(testDf),nrow(setboost))
setboost

###########################
###########################
# Warning!
###########################
###########################
# For boosting, we need to convert y back to numeric variable
###########################
trainDfB = trainDf # create a copy of train data, use this copy for boosting
trainDfB$y = as.numeric(trainDfB$y)-1 # convert factor labels to numeric variable
testDfB = testDf # create a copy of test data
testDfB$y = as.numeric(testDfB$y)-1



for(i in 1:nrow(setboost)) {
  ##fit and predict
  fboost = gbm(y~., data=trainDfB, distribution="bernoulli",
               n.trees=setboost[i,2],
               interaction.depth=setboost[i,1],
               shrinkage=setboost[i,3])
  
  phat = predict(fboost,
                 newdata=testDfB,
                 n.trees=setboost[i,2],
                 type="response")
  
  phatL$boost[,i] = phat
}




######################################################################
# Analysis of results
# Misclassification of Logistic regression on testing set
getConfusionMatrix(testDf$y, phatL[[1]][,1], 0.5)
cat('Missclassification rate = ', lossMR(testDf$y, phatL[[1]][,1], 0.5), '\n')


# random forest
nrun = nrow(setrf)
for(j in 1:nrun) {
  print(setrf[j,])
  print("Confusion Matrix:")
  print(getConfusionMatrix(testDf$y, phatL[[2]][,j], 0.5))
  cat('Missclassification rate = ', lossMR(testDf$y, phatL[[2]][,j], 0.5), '\n')
}


# boosting
nrun = nrow(setboost)
for(j in 1:nrun) {
  print(setboost[j,])
  print("Confusion Matrix:")
  print(getConfusionMatrix(testDf$y, phatL[[3]][,j], 0.5))
  cat('Missclassification rate = ', lossMR(testDf$y, phatL[[3]][,j], 0.5), '\n')
}



########################################################
# Loss on testing set
lossL = list()
nmethod = length(phatL)
for(i in 1:nmethod) {
  nrun = ncol(phatL[[i]])
  lvec = rep(0,nrun)
  for(j in 1:nrun) lvec[j] = lossf(testDf$y, phatL[[i]][,j])
  lossL[[i]]=lvec; names(lossL)[i] = names(phatL)[i]
}
lossv = unlist(lossL)
plot(lossv, ylab="loss on Test", type="n")
nloss=0
for(i in 1:nmethod) {
  ii = nloss + 1:ncol(phatL[[i]])
  points(ii,lossv[ii],col=i,pch=17)
  nloss = nloss + ncol(phatL[[i]])
}
legend("topright",legend=names(phatL),col=1:nmethod,pch=rep(17,nmethod))

# From each method class, we choose the one that has the lowest error on the validation set.

nmethod = length(phatL)
phatBest = matrix(0.0,nrow(testDf),nmethod) #pick off best from each method
colnames(phatBest) = names(phatL)
for(i in 1:nmethod) {
  nrun = ncol(phatL[[i]])
  lvec = rep(0,nrun)
  for(j in 1:nrun) lvec[j] = lossf(testDf$y,phatL[[i]][,j])
  imin = which.min(lvec)
  phatBest[,i] = phatL[[i]][,imin]
  phatBest[,i] = phatL[[i]][,1]
}

# We can plot $\hat p$ for best models on the test set
pairs(phatBest)


cost_benefit = matrix(c(0,-0.25,0,1), nrow=2)
print(cost_benefit)

confMat = getConfusionMatrix(testDf$y, phatBest[,1], 0.2)
print(confMat)
cat("Expected value of logistic regression = ", 
    sum(sum(confMat * cost_benefit)), "\n")

confMat = getConfusionMatrix(testDf$y, phatBest[,2], 0.2)
print(confMat)
cat("Expected value of random forests = ", 
    sum(sum(confMat * cost_benefit)), "\n")


confMat = getConfusionMatrix(testDf$y, phatBest[,3], 0.2)
print(confMat)
cat("Expected value of boosting = ", 
    sum(sum(confMat * cost_benefit)), "\n")


par(mfrow = c(1,1))
plot(c(0,1),c(0,1),xlab='False Positive Rate',ylab='True Positive Rate',main="ROC curve",cex.lab=1,type="n")
for(i in 1:ncol(phatBest)) {
  pred = prediction(phatBest[,i], testDf$y)
  # performance function calculates true positive rates and false positive rates
  perf = performance(pred, measure = "tpr", x.measure = "fpr")
  lines(perf@x.values[[1]], perf@y.values[[1]],col=i)
}
abline(0,1,lty=2)
legend("topleft",legend=names(phatL),col=1:nmethod,lty=rep(1,nmethod))

