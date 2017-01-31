library(kknn)
library(rpart)
library(rpart.plot)
library(tree)
library(randomForest)
library(gbm)
download.file("https://raw.githubusercontent.com/ChicagoBoothML/HelpR/master/docv.R", "docv.R")
source("docv.R")
#used.cars=read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/DATA___UsedCars/master/UsedCars.csv"))

########################### Nathan Code #################################
##############
# Homework 1 #
##############

########
# Load Config Files
########

options("width" = 250)
options(scipen = 999)
options(digits = 3)

library(ggplot2); require(gridExtra); library(MASS); library(Matrix)
library(kknn); library(boot); library(rpart); library(data.table)
set.seed(666) # the devils seed

xnaref <- function(x){
  if(is.factor(x))
    if(!is.na(levels(x)[1]))
      x <- factor(x,levels=c(NA,levels(x)),exclude=NULL)
    return(x) 
}

naref <- function(DF){
  if(is.null(dim(DF))) return(xnaref(DF))
  if(!is.data.frame(DF)) 
    stop("You need to give me a data.frame or a factor")
  DF <- lapply(DF, xnaref)
  return(as.data.frame(DF))
}

########
# Question 1
########

########
# Feature Engineering 
########

bike.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_train.csv")))
bike.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_test.csv")))

bike.train[,UID := .I]; setkey(bike.train, UID) # create UID column
bike.test[,UID := .I]; setkey(bike.test, UID) # create UID column
bike.test[,count := NA] # these are to be predicted

data <- rbind(bike.train, bike.test)	

data[ ,':='( # factor categorical vars
  year = (factor(year)),
  month = (factor(month)),
  day = (factor(day)),
  hour = (factor(hour)),
  season = (factor(season)),
  holiday = (factor(holiday)),
  workingday = (factor(workingday)),
  weather = (factor(weather))
)]

dummies <- model.matrix( ~ year + month + day + hour + season + holiday + workingday + weather + daylabel, data = data)[,-1]
all.data <- cbind(data[ ,c("count", "temp", "atemp", "humidity", "windspeed"), with = FALSE], dummies)

trainX <- Matrix(as.matrix(all.data[!is.na(count), !('count'), with = FALSE]))
trainY <- all.data[!is.na(count),  ('count'), with = FALSE]

testX <- Matrix(as.matrix(all.data[is.na(count), !('count'), with = FALSE]))
testY <- NULL

########
# Build Models
########
dt.bike.raw.train=data[!is.na(count)]
#I want to play with this data
dt.bike.raw.train$day_squared=dt.bike.raw.train$daylabel^2
dt.bike.raw.train$day_cubed=dt.bike.raw.train$daylabel^3
dt.bike.raw.train$atemp_squared=dt.bike.raw.train$atemp^2
dt.bike.raw.train$atemp_squareroot=dt.bike.raw.train$atemp^.5
dt.bike.raw.train$hour_numeric=as.numeric(dt.bike.raw.train$hour)
dt.bike.raw.train$humidity_squared=dt.bike.raw.train$humidity^2
dt.bike.raw.train$wind_squared=dt.bike.raw.train$windspeed^2
str(dt.bike.raw.train)
#end play with the data

sample.index=sample(nrow(dt.bike.raw.train),nrow(dt.bike.raw.train)/4)
dt.bike.test=dt.bike.raw.train[sample.index,]
dt.bike.train=dt.bike.raw.train[-sample.index,]

hist(dt.bike.train$count)
fit_base=lm(log(count+1)~poly(as.numeric(daylabel),3)+season+holiday+workingday+temp+atemp+windspeed,data=dt.bike.train)
summary(fit_base)

y_predict=predict(fit_base,newdata=dt.bike.test)
sqrt(sum((dt.bike.test$count-(exp(y_predict)+1))^2)/nrow(dt.bike.test))
#linear model suuucks


summary(dt.bike.train)
summary(dt.bike.train[count>500])

ggplot(dt.bike.train,aes(x=weather,y=count))+geom_boxplot()
ggplot(dt.bike.train,aes(x=as.factor(round(atemp,-1)),y=count))+geom_boxplot()
ggplot(dt.bike.train,aes(x=as.factor(round(humidity,-1)),y=count))+geom_boxplot()
ggplot(dt.bike.train,aes(x=as.factor(round(atemp,-1)),y=count))+geom_boxplot()


#Let's try a good knn model
y = as.vector(dt.bike.train$count)
x = dt.bike.train
x$count=4
x[,count:=NULL]
x[,year:=as.numeric(year)]
x[,month:=as.numeric(month)]
x[,day:=NULL]
x[,hour:=as.numeric(hour)]
x[,season:=as.numeric(season)]
x[,holiday:=as.numeric(holiday)]
x[,workingday:=as.numeric(workingday)]
x[,weather:=as.numeric(weather)]
mmsc=function(x) {return((x-min(x))/(max(x)-min(x)))}
xs = apply(x,2,mmsc) #apply scaling function to each column of x
kv = 1:10 #k values to try
cvtemp = docvknn(xs,y,kv,nfold=10)
cvtemp = sqrt(cvtemp/nrow(x)) #docvknn returns sum of squares
plot(kv,cvtemp)


#refit using all the data and k=4
ddf = data.frame(y,xs)
near5 = kknn(y~.,ddf,ddf,k=4,kernel = "rectangular")
lmf = lm(y~.,ddf)
fmat = cbind(y,near5$fitted,lmf$fitted)
colnames(fmat)=c("y","kNN5","linear")
pairs(fmat)
print(cor(fmat))

#knn RMSE
test.x = dt.bike.test
test.x$count=4
test.x[,count:=NULL]
test.x[,year:=as.numeric(year)]
test.x[,month:=as.numeric(month)]
test.x[,day:=NULL]
test.x[,hour:=as.numeric(hour)]
test.x[,season:=as.numeric(season)]
test.x[,holiday:=as.numeric(holiday)]
test.x[,workingday:=as.numeric(workingday)]
test.x[,weather:=as.numeric(weather)]
adjusted.test.x=data.frame(apply(test.x,2,mmsc))
kfbest=kknn(y~.,ddf,adjusted.test.x,k=4,kernel = "rectangular")
sqrt(sum((dt.bike.test$count-kfbest$fitted)^2)/nrow(dt.bike.test))

#ok, knn suuucks (108 error) - though maybe because got rid of factors?


#Let's try a tree
big.tree = rpart(count~., data=dt.bike.train, 
                 control=rpart.control(minsplit=5,  
                                       cp=0.0001,
                                       xval=10)   
)
nbig = length(unique(big.tree$where))
cat('size of big tree: ',nbig,'\n')

cptable = printcp(big.tree)
bestcp = cptable[ which.min(cptable[,"xerror"]), "CP" ]   # this is the optimal cp parameter

#plotcp(big.tree,ylim=c(0.05,.08)) # plot results
best.tree = prune(big.tree,cp=bestcp)
#rpart.plot(best.tree)

new.fit=predict(best.tree,newdata=dt.bike.test)
sqrt(sum((dt.bike.test$count-new.fit)^2)/nrow(dt.bike.test))

#How about a different method - let's use tree bagging
## bagging
y = as.vector(dt.bike.train$count)
x = dt.bike.train
x$count=4
x[,count:=NULL]
B=400
n=nrow(dt.bike.test)
nn = rep(0,B)
fmat=matrix(0,n,B)
set.seed(666)

#par(mfrow=c(1,2))
for(i in 1:B) {
  if((i%%100)==0) cat('i: ',i,'\n')
  ii = sample(1:n,n,replace=TRUE)
  nn[i] = length(unique(ii))
  bigtree = tree(count~.,dt.bike.train[ii,],mindev=.0002)
  #print(length(unique(bigtree$where)))
  temptree = prune.tree(bigtree,best=50)
  #print(length(unique(temptree$where)))
  fmat[,i]=predict(temptree,dt.bike.test)
}


par(mfrow=c(1,1))
#plot(dt.bike.train$x,ddf$y)
efit = apply(fmat,1,mean)
plot(efit,dt.bike.test$count)
sqrt(sum((efit-dt.bike.test$count)^2)/nrow(dt.bike.test))
#... depressingly worse - 93.6

#What about random forests?

p=ncol(dt.bike.train)-1
mtryv = c(p,sqrt(p))
ntreev = c(100,500)
parmrf = expand.grid(mtryv,ntreev)
colnames(parmrf)=c('mtry','ntree')
nset = nrow(parmrf)
olrf = rep(0,nset)
ilrf = rep(0,nset)
rffitv = vector('list',nset)
for(i in 1:nset) {
  temprf = randomForest(count~.,data=dt.bike.train,mtry=parmrf[i,1],ntree=parmrf[i,2])
  ifit = predict(temprf)
  ofit= predict(temprf,newdata=dt.bike.test)
  olrf[i] = sum((dt.bike.test$count-ofit)^2)
  ilrf[i] = sum((dt.bike.train$count-ifit)^2)
  rffitv[[i]]=temprf
}
ilrf = round(sqrt(ilrf/nrow(dt.bike.train)),3); olrf = round(sqrt(olrf/nrow(dt.bike.test)),3)
#----------------------------------------
#print losses
print(cbind(parmrf,olrf,ilrf))

#write validation predictions
iirf=which.min(olrf)
therf = rffitv[[iirf]]
therfpred=predict(therf,newdata=dt.bike.train)

#Wow, this is really good. When mtry is 21, olrf is 46
#Let's experiment with it a bit more
mrtys=4:7*5
ooserror=rep(0,length(mrtys))
iserror=rep(0,length(mrtys))
start_time=proc.time()
for(i in 1:length(mrtys))
{
  experimentrf = randomForest(count~.,mtry=mrtys[i],data=dt.bike.train,ntree=100)
  infit = predict(experimentrf)
  outfit= predict(experimentrf,newdata=dt.bike.test)
  ooserror[i]=sqrt(sum((dt.bike.test$count-outfit)^2)/nrow(dt.bike.test))
  iserror[i]=sqrt(sum((dt.bike.train$count-infit)^2)/nrow(dt.bike.train))
  print(paste0("done with ",i," of ",length(mrtys)))
  elapsed_time = proc.time() - start_time 
  print(elapsed_time[3])
}

cbind(mrtys,ooserror,iserror)

#Conclusion: high mtry is better (I got best with 30 (got 46.15 oos))

#Finally, let's try boosting!
idv = c(4, 10)
ntv = c(1000, 5000)
lamv=c(.001, .2)
parmb = expand.grid(idv,ntv,lamv)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)

nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)
for(i in 1:nset) {
  tempboost = gbm(count~.,data=dt.bike.train,distribution='gaussian',
                  interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
  ifit = predict(tempboost,n.trees=parmb[i,2])
  ofit=predict(tempboost,newdata=dt.bike.test,n.trees=parmb[i,2])
  olb[i] = sum((dt.bike.test$count-ofit)^2)
  ilb[i] = sum((dt.bike.train$count-ifit)^2)
  bfitv[[i]]=tempboost
  print(paste0("done with ",i," of ",nset))
}
ilb = round(sqrt(ilb/nrow(dt.bike.train)),3); 
olb = round(sqrt(olb/nrow(dt.bike.test)),3)

print(cbind(parmb,olb,ilb))

#Let's hone in a little more
idv = c(10,20)
ntv = c(2000,5000)
lamv=c(.02,.03)
parmb = expand.grid(idv,ntv,lamv)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)

nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)
start_time=proc.time()
for(i in 1:nset) {
  tempboost = gbm(count~.,data=dt.bike.train,distribution='gaussian',
                  interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
  ifit = predict(tempboost,n.trees=parmb[i,2])
  ofit=predict(tempboost,newdata=dt.bike.test,n.trees=parmb[i,2])
  olb[i] = sum((dt.bike.test$count-ofit)^2)
  ilb[i] = sum((dt.bike.train$count-ifit)^2)
  bfitv[[i]]=tempboost
  print(paste0("done with ",i," of ",nset))
  elapsed_time = proc.time() - start_time 
  print(elapsed_time[3])
}
ilb = round(sqrt(ilb/nrow(dt.bike.train)),3); 
olb = round(sqrt(olb/nrow(dt.bike.test)),3)

print(cbind(parmb,olb,ilb))
#best results so far - tdepth 20 (gives about 1 over 10- but mixed - running with 10 is much faster), ntree 5000, lam of .1 (about 3 better than .2)
#even better results: 39 (20 5000 lam-.02)
#Better with lam of .05

#write validation predictions
iib=which.min(olb)
theb = bfitv[[iib]] 
thebpred = predict(theb,newdata=caval,n.trees=parmb[iib,2])


#Highlights:
#Way more big days in summer (particularly the edges - 6,9,5), way more in 2012 (NEED A TIME VARIABLE THAT ISN"T A FACTOR)
#Big times - hour 17,18,8,19
#Chris says he can get 42
#
#
########
# Question 2
########

########
# Feature Engineering 
########

movies.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/MovieReview_train.csv")))
movies.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/MovieReview_test.csv")))

#movies.train[,UID := .I]; setkey(movies.train, UID) # create UID column
#movies.test[,UID := .I]; setkey(movies.test, UID) # create UID column
#movies.test[,sentiment := NA] # these are to be predicted

#all.data <- rbind(movies.train, movies.test)	

#trainX <- Matrix(as.matrix(all.data[!is.na(sentiment), !('sentiment'), with = FALSE]))
#trainY <- all.data[!is.na(sentiment), ('sentiment'), with = FALSE]

#testX <- Matrix(as.matrix(all.data[is.na(sentiment), !('sentiment'), with = FALSE]))
#testY <- NULL

########
# Build Models 
########


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

getConfusionMatrix = function(y,phat,thr=0.5) {
  if(is.factor(y)) y = as.numeric(y)-1
  yhat = ifelse(phat > thr, 1, 0)
  tb = table(predictions = yhat, 
             actual = y)  
  rownames(tb) = c("predict_0", "predict_1")
  return(tb)
}

lossMR = function(y,phat,thr=0.5) {
  if(is.factor(y)) y = as.numeric(y)-1
  yhat = ifelse(phat > thr, 1, 0)
  return(1 - mean(yhat == y))
}

#Clean the data a bit more
sample.index=sample(nrow(movies.train),nrow(movies.train)/4)
dt.movies.test=movies.train[sample.index,]
dt.movies.train=movies.train[-sample.index,]



lgfit = glm(sentiment~., dt.movies.train, family=binomial)
print(summary(lgfit))

temp.predict=predict(lgfit,newdata=dt.movies.test)
lossMR(dt.movies.test$sentiment,temp.predict)
#20% Misclassification rate

#maybe a gamlr?
dt.movies.train.without.sent=dt.movies.train
dt.movies.train.without.sent$sentiment=0
dt.movies.test.without.sent=dt.movies.test
dt.movies.test.without.sent$sentiment=0
temp=cv.gamlr(dt.movies.train.without.sent,dt.movies.train$sentiment, family = 'binomial')
plot(temp)
coef(temp)
newpredict=predict(temp,newdata=dt.movies.test)
lossMR(dt.movies.test$sentiment,newpredict)
#equally sucks - .21
dt.test=data.table(c(1,2,3),c(2,3,4))
dt.test[,-1]

#Random Forest!
p=ncol(dt.movies.train)-1
mtryv = c(p/2, sqrt(p)/2)
ntreev = c(100,5000)
(setrf = expand.grid(mtryv,ntreev))  # this contains all settings to try
colnames(setrf)=c("mtry","ntree")
phatL = matrix(0.0,nrow(dt.movies.test),nrow(setrf))  # we will store results here

###fit rf
for(i in 1:nrow(setrf)) {
  #fit and predict
  frf = randomForest(as.factor(sentiment)~., data=dt.movies.train, 
                     mtry=setrf[i,1],
                     ntree=setrf[i,2],
                     nodesize=10)
  phat = predict(frf, newdata=dt.movies.test, type="prob")[,2]
  print(lossMR(dt.movies.test$sentiment,phat))
  phatL[,i]=phat
}
lossMR(dt.movies.test$sentiment,phat)
#The one one I tried sucked - 23% - slow to run


#aaaand boosting
##settings for boosting
idv = c(2,4)
ntv = c(1000,5000)
shv = c(.1,.01)
(setboost = expand.grid(idv,ntv,shv))
colnames(setboost) = c("tdepth","ntree","shrink")
phatL$boost = matrix(0.0,nrow(dt.movies.test),nrow(setboost))
for(i in 1:nrow(setboost)) {
  ##fit and predict
  fboost = gbm(sentiment~., data=dt.movies.train, distribution="bernoulli",
               n.trees=setboost[i,2],
               interaction.depth=setboost[i,1],
               shrinkage=setboost[i,3])
  
  phat = predict(fboost,
                 newdata=dt.movies.test,
                 n.trees=setboost[i,2],
                 type="response")
  
  phatL$boost[,i] = phat
  print(lossMR(dt.movies.test$sentiment,phat))
}

lossMR(dt.movies.test$sentiment,phat)

#Boosting is fine, but not excellent? 20.4% - ranged from .204 to .208

#Let's try his variable importance analysis for boosting:


boostfit = gbm(sentiment~.,data=dt.movies.train,
               distribution='bernoulli',
               interaction.depth=4,
               n.trees=500,
               shrinkage=.2)

p=ncol(accidents_df_train)-1
vsum=summary(boostfit, plotit=F) #this will have the variable importance info

#write variable importance table
print(vsum)

#plot variable importance
#the package does this automatically, but I did not like the plot
plot(vsum$rel.inf,axes=F,pch=16,col='red')
axis(1,labels=vsum$var,at=1:p)
axis(2)
for(i in 1:p) lines(c(i,i),c(0,vsum$rel.inf[i]),lwd=4,col='blue')

b_test_predictions = predict(boostfit, dt.movies.test, n.trees = 500, type = "response") 

class0_ind =  b_test_predictions < 0.5  
class1_ind =  b_test_predictions >= 0.5 
b_test_predictions[class0_ind] = 0
b_test_predictions[class1_ind] = 1
(1 - mean(b_test_predictions == dt.movies.test$sentiment))  

tb_rf = table(predictions = b_test_predictions, 
              actual = dt.movies.test$sentiment)  
rownames(tb_rf) = c("predict_0_sentiment", "predict_1_sentiment")
print(tb_rf)

dt.movies.train_vs = dt.movies.train



keeps = c(rownames(vsum[vsum$rel.inf>.1,]),"sentiment")
dt.movies.train_vs = dt.movies.train[, (names(dt.movies.train) %in% keeps),with=FALSE]

boostfit = gbm(sentiment~.,data=dt.movies.train_vs,
               distribution='bernoulli',
               interaction.depth=4,
               n.trees=500,
               shrinkage=.2)


dt.movies.test_vs=dt.movies.test[, (names(dt.movies.test) %in% keeps),with=FALSE]
phat = predict(boostfit,
               newdata=dt.movies.test_vs,
               n.trees=500,
               type="response")

print(lossMR(dt.movies.test$sentiment,phat))


#That didn't work at all either!


#Naive Bayes
library(RTextTools)
library(e1071)
?naiveBayes

classifier=naiveBayes(dt.movies.train.without.sent,as.factor(dt.movies.train$sentiment))
predicted=predict(classifier,dt.movies.test)
table(dt.movies.test$sentiment,predicted)
recall_accuracy(dt.movies.test$sentiment,predicted)

#Let's do it the way he does it

nb_model = naiveBayes(as.factor(sentiment) ~ ., dt.movies.train)
nb_test_predictions = predict(nb_model, dt.movies.test) 
table(dt.movies.test$sentiment,nb_test_predictions)

1 - mean(nb_test_predictions == dt.movies.test$sentiment)
tb_tree = table(predictions = nb_test_predictions, 
                actual = dt.movies.test$sentiment)  
rownames(tb_tree) = c("predict_NO_INJURY", "predict_INJURY")
print(tb_tree)

#damn, that actually stinks - 23.7%


