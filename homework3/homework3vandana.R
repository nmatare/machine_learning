---
title: "Homework 3"
author: "Vandana"
date: "February 24, 2017"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(cache = TRUE)
options(digits=3)
options(width = 48)
```


#1.1

Load the data

```{r}
if("R.utils" %in% rownames(installed.packages()) == FALSE) {install.packages("R.utils")}
if("data.table" %in% rownames(installed.packages()) == FALSE) {install.packages("data.table")}
library("R.utils")
library("data.table")
gitURL="https://github.com/ChicagoBoothML/MLClassData/raw/master/KDDCup2009_Customer_relationship/";
DownloadFileList=c("orange_small_train.data.gz","orange_small_train_appetency.labels.txt",
                   "orange_small_train_churn.labels.txt","orange_small_train_upselling.labels.txt")
LoadFileList=c("orange_small_train.data","orange_small_train_appetency.labels.txt",
               "orange_small_train_churn.labels.txt","orange_small_train_upselling.labels.txt")
for (i in 1:length(LoadFileList)){
  if (!file.exists(LoadFileList[[i]])){
    if (LoadFileList[[i]]!=DownloadFileList[[i]]) {
      download.file(paste(gitURL,DownloadFileList[[i]],sep=""),destfile=DownloadFileList[[i]])
      gunzip(DownloadFileList[[i]])
    }else{
      download.file(paste(gitURL,DownloadFileList[[i]],sep=""),destfile=DownloadFileList[[i]])}}
}
na_strings <- c('',
                'na', 'n.a', 'n.a.',
                'nan', 'n.a.n', 'n.a.n.',
                'NA', 'N.A', 'N.A.',
                'NaN', 'N.a.N', 'N.a.N.',
                'NAN', 'N.A.N', 'N.A.N.',
                'nil', 'Nil', 'NIL',
                'null', 'Null', 'NULL')
X=as.data.table(read.table('orange_small_train.data',header=TRUE,
                           sep='\t', stringsAsFactors=TRUE, na.strings=na_strings))
Y_churn =read.table("orange_small_train_churn.labels.txt", quote="\"")
Y_appetency=read.table("orange_small_train_appetency.labels.txt", quote="\"")
Y_upselling=read.table("orange_small_train_upselling.labels.txt", quote="\"")
```

#1.2

To clean the data, first let us see which columns have more than half N/As

```{r}
nvar = length(X)
meanNA = matrix(0.0,1,nvar)
colnames(meanNA) = names(X)
for (i in names(X)) {
  CurrentColumn = X[[i]]
  CurrentColumnVariableName = i
  meanNA[,i] = mean(is.na(CurrentColumn))
  cat(i, mean(is.na(CurrentColumn)), "\n")
}
```

Next, we remove columns with more than half N/As
```{r}
Columnstoexclude = which(meanNA>.5)
XS = X[, -Columnstoexclude, with = FALSE]
```

We then replace column means for N/As in numeric columns
```{r}
nvar = length(XS)
numericCOL = matrix(0.0,1,nvar)
colnames(numericCOL) = names(XS)
for (i in names(XS)) {
  CurrentColumn = X[[i]]
  CurrentColumnVariableName = i
  numericCOL[,i] = is.numeric(CurrentColumn)
}

nCOLindex = which(numericCOL==1)

for (i in nCOLindex) {
meanofCOL = mean(XS[[i]],na.rm = TRUE)
CurrentColumn = XS[[i]] 
idx = is.na(CurrentColumn) 
CurrentColumn[idx] = meanofCOL
XS[,i] = CurrentColumn 
}
```

Next, Replace N/As in factor columns with new factor
```{r}
nCOLindex = which(numericCOL==0)

for (i in nCOLindex) {
CurrentColumn = XS[[i]] 
CurrentColumnName = names(XS)[i]
idx = is.na(CurrentColumn) 
CurrentColumn = as.character(CurrentColumn) 
CurrentColumn[idx] = paste(CurrentColumnName, "_NA", sep = "") 
CurrentColumn = as.factor(CurrentColumn) 
XS[[i]] = CurrentColumn 
}
```

Aggregate a many lower frequency factors into new factors
```{r}
Thres_Low = 249
Thres_Medium = 499
Thres_High = 999

for (i in nCOLindex) {
CurrentColumn = XS[[i]] 
CurrentColumnName = names(XS)[i]
CurrentColumn_Table = table(CurrentColumn) 
levels(CurrentColumn)[CurrentColumn_Table <= Thres_Low] = paste(CurrentColumnName, "_Low", sep = "")
CurrentColumn_Table2 = table(CurrentColumn)
levels(CurrentColumn)[CurrentColumn_Table2 > Thres_Low & CurrentColumn_Table2 <= Thres_Medium] = paste(CurrentColumnName, "_Medium", sep = "")
CurrentColumn_Table3 = table(CurrentColumn)
levels(CurrentColumn)[CurrentColumn_Table3 > Thres_Medium & CurrentColumn_Table3 <= Thres_High] = paste(CurrentColumnName, "_High", sep = "")
XS[[i]] = CurrentColumn #Plug-back to the data.frame
}
```

Delete the column in XS that only has one level
```{r}
XS$Var202 = NULL
```

Make the Y's factors 0/1
```{r}
Y_churn$V1[Y_churn$V1=="-1"] = 0
Y_appetency$V1[Y_appetency$V1=="-1"] = 0
Y_upselling$V1[Y_upselling$V1=="-1"] = 0
Y_churn = as.factor(Y_churn$V1)
Y_churn = as.data.frame(Y_churn)
Y_appetency = as.factor(Y_appetency$V1)
Y_appetency = as.data.frame(Y_appetency)
Y_upselling = as.factor(Y_upselling$V1)
Y_upselling = as.data.frame(Y_upselling)
```

Split the data into 40% Train, 40% Validation and 20% Test
```{r}
set.seed(666)
n = nrow(XS)

n1=floor(n/5)*2
n2=floor(n/5)*2
n3=n-n1-n2
ii = sample(1:n,n)


XS_train = XS[ii[1:n1],]
Y_appetency_train = as.data.frame(Y_appetency[ii[1:n1],])
colnames(Y_appetency_train)[1] = "V1"
Y_churn_train = as.data.frame(Y_churn[ii[1:n1],])
colnames(Y_churn_train)[1] = "V1"
Y_upselling_train = as.data.frame(Y_upselling[ii[1:n1],])
colnames(Y_upselling_train)[1] = "V1"
XS_validation = XS[ii[n1+1:n2],]
Y_appetency_validation = as.data.frame(Y_appetency[ii[n1+1:n2],])
colnames(Y_appetency_validation)[1] = "V1"
Y_churn_validation = as.data.frame(Y_churn[ii[n1+1:n2],])
colnames(Y_churn_validation)[1] = "V1"
Y_upselling_validation = as.data.frame(Y_upselling[ii[n1+1:n2],])
colnames(Y_upselling_validation)[1] = "V1"
XS_test = XS[ii[n1+n2+1:n3],]
Y_appetency_test = as.data.frame(Y_appetency[ii[n1+n2+1:n3],])
colnames(Y_appetency_test)[1] = "V1"
Y_churn_test = as.data.frame(Y_churn[ii[n1+n2+1:n3],])
colnames(Y_churn_test)[1] = "V1"
Y_upselling_test = as.data.frame(Y_upselling[ii[n1+n2+1:n3],])
colnames(Y_upselling_test)[1] = "V1"
```

Create functions that will be good to compare classifier models

1. Deviance loss function
```{r}
lossf = function(y,phat,wht=0.0000001) {
  if(is.factor(y)) y = as.numeric(y)-1
  phat = (1-wht)*phat + wht*.5
  py = ifelse(y==1, phat, 1-phat)
  return(-2*sum(log(py)))
}
```

2. ConfusionMatrix function
```{r}
getConfusionMatrix = function(y,phat,thr=0.5) {
  if(is.factor(y)) y = as.numeric(y)-1
  yhat = ifelse(phat > thr, 1, 0)
  tb = table(predictions = yhat,
             actual = y)
  rownames(tb) = c("predict_0", "predict_1")
  return(tb)
}
```

3. Misclassification rate function
```{r}
lossMR = function(y,phat,thr=0.5) {
  if(is.factor(y)) y = as.numeric(y)-1
  yhat = ifelse(phat > thr, 1, 0)
  return(1 - mean(yhat == y))
}
```

See the split of 0/1s on the Ys
```{r}
(table(Y_appetency))
(table(Y_churn))
(table(Y_upselling))

(table(Y_appetency_train))
(table(Y_churn_train))
(table(Y_upselling_train))

(table(Y_appetency_validation))
(table(Y_churn_validation))
(table(Y_upselling_validation))

(table(Y_appetency_test))
(table(Y_churn_test))
(table(Y_upselling_test))
```

#1.3

Store the test phat for the different methods here
```{r}
phatL = list()
```

We chose to do all the analysis on the upselling variable. First we looked into fitting classifiers with only one explanatory variable.
```{r}
p = length(XS_train)
phatL$logit = matrix(0.0,nrow(XS_validation),p) # we will store results here

for(i in 1:p) {
  #fit and predict
  lgfit = glm(Y_upselling_train$V1 ~ XS_train[[i]],family = binomial)
  phat = predict(lgfit, XS_validation, type="response")
  phatL$logit[,i]=phat
}

lossL = list()
nmethod = length(phatL)
for(i in 1:nmethod) {
  nrun = ncol(phatL[[i]])
  lvec = rep(0,nrun)
  for(j in 1:nrun) lvec[j] = lossf(Y_upselling_validation, phatL[[i]][,j])
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
```

As can be seen from the graph, variable selection is difficult using this approach.
Next we attempted to use Random Forests to see variable importance and select variables.
Using random forests to reduce variables
```{r}
library(randomForest)
XS_train_rf = XS_train
XS_train_rf$upselling = Y_upselling_train$V1
XS_validation_rf = XS_validation
XS_validation_rf$upselling = Y_upselling_validation$V1

p=ncol(XS_train_rf)-1
mtryv = c(p, round(sqrt(p)))
ntreev = c(500,1000)
setrf = expand.grid(mtryv,ntreev) # this contains all settings to try
colnames(setrf)=c("mtry","ntree")
phatL$rf = matrix(0.0,nrow(XS_validation_rf),nrow(setrf)) # we will store results here
```

Fit rf
```{r}
for(i in 1:nrow(setrf)) {
  #fit and predict
  frf = randomForest(upselling~., data=XS_train_rf,
                     mtry=setrf[i,1],
                     ntree=setrf[i,2],
                     nodesize=10)
  phat = predict(frf, newdata=XS_validation_rf, type="prob")[,2]
  phatL$rf[,i]=phat
}
```

Plot Deviance Loss
```{r}
lossL = list()
nmethod = length(phatL)
for(i in 1:nmethod) {
  nrun = ncol(phatL[[i]])
  lvec = rep(0,nrun)
  for(j in 1:nrun) lvec[j] = lossf(Y_upselling_validation, phatL[[i]][,j])
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
legend("bottomleft",legend=names(phatL),col=1:nmethod,pch=rep(17,nmethod))
```

The graph shows that the performance of these models are much better than the single independent variable linear classifiers

```{r}
(lossf(Y_upselling_validation, phatL[[2]][,1]))
(lossf(Y_upselling_validation, phatL[[2]][,2]))
(lossf(Y_upselling_validation, phatL[[2]][,3]))
(lossf(Y_upselling_validation, phatL[[2]][,4]))
```

The best model when looking at deviance is the model with p = 8 and 1000 trees
Lets rerun this model to get the variable importance

```{r}
rforest1 = randomForest(upselling~., data=XS_train_rf, mtry=8,ntree=1000,nodesize=10, importance=T)
```
plot variable importance
```{r}
varImpPlot(rforest1)
```

sort the variables by importance
```{r}
ImpVar = as.data.frame(importance(rforest1))
```

How many variables have a MeanDecreaseAccurary greater than 20?
```{r}
sum(ImpVar$MeanDecreaseAccuracy>20)
```

These are the most important variables and as a result will be the variables we will keep

```{r}
ImpVarKeep = ImpVar[ImpVar$MeanDecreaseAccuracy>20,]
ImpVarKeep = as.data.frame(t(ImpVarKeep))
(ImpVarKeepNames = names(ImpVarKeep))

XS_train = XS_train[ImpVarKeepNames]
XS_validation = XS_validation[ImpVarKeepNames]
XS_test = XS_test[ImpVarKeepNames]
```

#1.4

Combine the training and validation sets
```{r}
XS_trainVal = rbind(XS_train,XS_validation)
Y_appetency_trainVal = rbind(Y_appetency_train,Y_appetency_validation)
Y_churn_trainVal = rbind(Y_churn_train,Y_churn_validation)
Y_upselling_trainVal = rbind(Y_upselling_train,Y_upselling_validation)
colnames(Y_upselling_trainVal) = "upselling"
colnames(Y_upselling_test) = "upselling"
```

Now we fit random forests
```{r}
XS_train_rf2 = XS_trainVal
XS_train_rf2$upselling = Y_upselling_trainVal$V1

p=ncol(XS_train_rf2)-1
rforest2 = randomForest(upselling~., data=XS_train_rf2, mtry=round(sqrt(p)), ntree=1000, nodesize=10)
```

Lets look at the Out-of-bag error plot to see what the best number of trees should be
```{r}
plot(rforest2$err.rate[,"OOB"], xlab="# trees", ylab="OOB error", cex=0.3)

(optimaltrees = which.min(rforest2$err.rate[,"OOB"]))

cat('It looks like the best random forest has trees = ', optimaltrees, '\n')
```

Refit the final random forest
```{r}
rffinal = randomForest(upselling~., data=XS_train_rf2,mtry=round(sqrt(p)),ntree=optimaltrees,nodesize=10)

phat = predict(rffinal, newdata=XS_train_rf2, type="prob")[,2]
```

Lets look at the in sample misclassification rates and deviance of the model

```{r}
print("Confusion Matrix:")
print(getConfusionMatrix(XS_train_rf2$upselling, phat, 0.5))
cat('Missclassification rate = ', lossMR(XS_train_rf2$upselling, phat, 0.5), '\n')
cat('Deviance Loss = ', lossf(XS_train_rf2$upselling, phat), '\n')
```


Let's see how boosted trees perform:
```{r}
# load boosted trees
```


#1.5

Lets look at the out of sample misclassification rates and deviance of the model tested on the test set

```{r}
phatOOS = predict(rffinal, newdata=XS_test, type="prob")[,2]


print("Confusion Matrix:")
print(getConfusionMatrix(Y_upselling_test$upselling, phatOOS, 0.5))
cat('Missclassification rate = ', lossMR(Y_upselling_test$upselling, phatOOS, 0.5), '\n')
cat('Deviance Loss = ', lossf(Y_upselling_test$upselling, phatOOS), '\n')
```

#Question 2

Downloading and cleaning the data
```{r}
parse_human_activity_recog_data <- function(
  data_path='https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/HumanActivityRecognitionUsingSmartphones',
  X_names_file_name='features.txt',
  train_subfolder_name='train', X_train_file_name='X_train.txt', y_train_file_name='y_train.txt',
  test_subfolder_name='test', X_test_file_name='X_test.txt', y_test_file_name='y_test.txt') {
  
  library(data.table)
  
  cat('Parsing Data Set "UCI Human Activity Recognition Using Smartphones"...\n')
  
  cat("   Parsing Unique Input Features' (X's) Names... ")
  X_names_with_duplicates <- fread(
    file.path(data_path, X_names_file_name),
    header=FALSE,
    drop=c(1),
    showProgress=FALSE)$V2
  X_unique_names <- sort(unique(X_names_with_duplicates))
  cat('done!\n')
  
  cat('   Parsing Train & Test Input Feature Data Sets... ')
  X_train_with_duplicates <- fread(
    file.path(data_path, train_subfolder_name, X_train_file_name),
    header=FALSE,
    col.names=X_names_with_duplicates,
    colClasses='numeric',
    showProgress=FALSE)
  X_train <- X_train_with_duplicates[ , X_unique_names, with=FALSE]
  X_test_with_duplicates <- fread(
    file.path(data_path, test_subfolder_name, X_test_file_name),
    header=FALSE,
    col.names=X_names_with_duplicates,
    colClasses='numeric',
    showProgress=FALSE)
  X_test <- X_test_with_duplicates[ , X_unique_names, with=FALSE]
  cat('done!\n')
  
  cat('   Parsing Train & Test Labels (y)... ')
  y_class_labels <- c(
    'Walking', 'WalkingUpstairs', 'WalkingDownstairs', 'Sitting', 'Standing', 'Laying')
  
  y_train = factor(
    fread(
      file.path(data_path, train_subfolder_name, y_train_file_name),
      header=FALSE)$V1,
    levels=1 : 6,
    labels=y_class_labels)
  
  y_test = factor(
    fread(
      file.path(data_path, test_subfolder_name, y_test_file_name),
      header=FALSE)$V1,
    levels=1 : 6,
    labels=y_class_labels)
  cat('done!\n')
  
  cat('   Removing Data Rows with Missing (NaN) Values... ')
  train_rows_not_na_yesno <- !is.na(rowSums(X_train))
  X_train <- X_train[train_rows_not_na_yesno, ]
  y_train <- y_train[train_rows_not_na_yesno]
  test_rows_not_na_yesno <- !is.na(rowSums(X_test))
  X_test <- X_test[test_rows_not_na_yesno]
  y_test = y_test[test_rows_not_na_yesno]
  cat('done!\n')
  
  list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
}

download.file(
  paste("https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/",
        "HumanActivityRecognitionUsingSmartphones/ParseData.R",sep=""),
  "ParseData.R")
data <- parse_human_activity_recog_data()

dt.train.x=data.table(data$X_train)
dt.train.y=data.table(data$y_train)
dt.test.x=data.table(data$X_test)
dt.test.y=data.table(data$y_test)

dt.train.y


library(h2o)
library(rgl)

h2oServer <- h2o.init(nthreads=-1)

dfh2o = as.h2o(data.frame(x=dt.train.x,y=dt.train.y), destination_frame = "human.train")
dftest = as.h2o(data.frame(x=dt.test.x,y=dt.test.y), destination_frame = "human.test")
dim(dt.train.x)
```

1 hidden 2 neurons
```{r}
model = h2o.deeplearning( y=y,training_frame = dfh2o,
                          hidden = 2,
                          epochs = 1000,
                          export_weights_and_biases = TRUE,
                          model_id = "model.h1n2",
                          seed = 1
)
phat = predict(model, dftest)
phatL$h1n10 = as.matrix( phat[,3] )
```

We need to load data into h2o format
```{r}
train_hex = as.h2o(data.frame(x=dt.train.x,y=dt.train.y), destination_frame = "human.train")
test_hex = as.h2o(data.frame(x=dt.test.x,y=dt.test.y), destination_frame = "human.test")

predictors = 1:477
response = 478

train_hex[,response] <- as.factor(train_hex[,response])
test_hex[,response] <- as.factor(test_hex[,response])
```

Create frames with input features only
We will need these later for unsupervised training
```{r}
trainX = train_hex[,-response]
testX = test_hex[,-response]
```


Did this with 10 epochs and 100.  100 gives you lower train error, does nothing for test
Test about 6% with this
```{r}
dl_model <- h2o.deeplearning(x=predictors, y=response,
                             training_frame=train_hex,
                             epochs=100
)
h2o.saveModel(dl_model, path = "mnist" )  
```

Performance on test
```{r}
ptest = h2o.performance(dl_model, test_hex )
h2o.confusionMatrix(ptest)
```

Performance on train
```{r}
ptrain = h2o.performance(dl_model, train_hex)
h2o.confusionMatrix(ptrain)
```

Training many models to see which may do well
```{r}
bigtest=FALSE
if (bigtest == FALSE) {
  # it will take some time to train all
  # Did this with 2 epochs and 20.  As above, affects train but not really test
  # Smallest error should be about 5% test
  EPOCHS = 20
  args <- list(
    list(epochs=EPOCHS),
    list(epochs=EPOCHS, activation="Tanh"),
    list(epochs=EPOCHS, hidden=c(512,512)),
    list(epochs=5*EPOCHS, hidden=c(64,128,128)),
    list(epochs=5*EPOCHS, hidden=c(512,512), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(256,256,256), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(200,200), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(100,100,100), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5)
  )
  
  run <- function(extra_params) {
    str(extra_params)
    print("Training.")
    model <- do.call(h2o.deeplearning, modifyList(list(x=predictors, y=response,
                                                       training_frame=train_hex), extra_params))
    sampleshist <- model@model$scoring_history$samples
    samples <- sampleshist[length(sampleshist)]
    time <- model@model$run_time/1000
    print(paste0("training samples: ", samples))
    print(paste0("training time   : ", time, " seconds"))
    print(paste0("training speed  : ", samples/time, " samples/second"))
    
    print("Scoring on test set.")
    p <- h2o.performance(model, test_hex)
    cm <- h2o.confusionMatrix(p)
    test_error <- cm$Error[length(cm$Error)]
    print(paste0("test set error  : ", test_error))
    
    c(paste(names(extra_params), extra_params, sep = "=", collapse=" "), 
      samples, sprintf("%.3f", time), 
      sprintf("%.3f", samples/time), sprintf("%.3f", test_error))
  }
  
  writecsv <- function(results) {
    table <- matrix(unlist(results), ncol = 5, byrow = TRUE)
    colnames(table) <- c("parameters", "training samples",
                         "training time", "training speed", "test set error")
    table
  }
  
  table = writecsv(lapply(args, run))
  save(table, file="mnist.h2o.table_results.RData")
  
} else {
  load("mnist.h2o.table_results.RData")
  table
}
```

#2.2
```{r}
library(tree)
library(randomForest)
num_trees = 500
```

This is a little worse than DNNs above, something like 6% test error
```{r}
rf_28 = randomForest(
  x=dt.train.x,
  y=as.factor(dt.train.y$V1), 
  xtests=dt.test.x,
  sampsize=6000,   # sample about 10% of data
  ntree=num_trees, 
  mtry=22,         # try 22 = sqrt(477) features at each split
  importance=TRUE, 
  nodesize=20     # need this many observations in the leaf
)

rf_28

varImpPlot(rf_28, type=2, n.var=20, main="Variable importance")
predicted.test = predict(rf_28, dt.test.x)
table(predicted.test,dt.test.y$V1)
sum(predicted.test!=dt.test.y$V1)/nrow(dt.test.y)
```
