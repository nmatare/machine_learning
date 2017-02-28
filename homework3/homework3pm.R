##################### Getting the Data#######################

PackageList =c('MASS','gbm','tree','randomForest','rpart','caret','ROCR','readxl','data.table','R.utils') 
NewPackages=PackageList[!(PackageList %in% 
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE)#array function


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

X=as.data.table(read.table('orange_small_train.data.gz',header=TRUE,
                           sep='\t', stringsAsFactors=TRUE, na.strings=na_strings))
Y_churn    =read.table("orange_small_train_churn.labels.txt", quote="\"")
Y_appetency=read.table("orange_small_train_appetency.labels.txt", quote="\"")
Y_upselling=read.table("orange_small_train_upselling.labels.txt", quote="\"")




i="Var220"


#1. How to write a loop over columns of a data.frame (say, X here)?

ExcludeVars=array()
j=1
for (i in names(X)){
  add=FALSE #a dummy variable to see if we go up a count
  CurrentColumn=X[[i]]
  CurrentColumnVariableName=i
  if(mean(is.na(CurrentColumn))>.5){ExcludeVars[j]=CurrentColumnVariableName;add=TRUE} #majority straight up NAs
  #1 value other than na
  if(length(unique(CurrentColumn[!is.na(CurrentColumn)]))==1){ExcludeVars[j]=CurrentColumnVariableName;add=TRUE}
  #Now let's deal with numeric nas
  if(is.numeric(CurrentColumn))
  {
    idx=is.na(CurrentColumn)
    CurrentColumn[idx]=mean(CurrentColumn,na.rm=TRUE)
    X[[i]]=CurrentColumn
  }
  #And now factor nas
  if(is.factor(CurrentColumn))
  {
    idx=is.na(CurrentColumn)                 #Locate the NAs
    CurrentColumn=as.character(CurrentColumn)#Convert from factor to characters
    CurrentColumn[idx]=paste(i,'_NA',sep="") #Add the new NA level strings
    CurrentColumn=as.factor(CurrentColumn)   #Convert back to factors
    X[[i]]=CurrentColumn  
    
    #And now aggregate factors into larger factors
    Thres_Low=249;
    Thres_Medium=499;
    Thres_High=999;
    CurrentColumn_Table=table(CurrentColumn) #Tabulate the frequency
    levels(CurrentColumn)[CurrentColumn_Table<=Thres_Low]=paste(i,'_Low',sep="")
    CurrentColumn_Table=table(CurrentColumn)
    levels(CurrentColumn)[CurrentColumn_Table>Thres_Low & CurrentColumn_Table<=Thres_Medium]=paste(i,'_Medium',sep="")
    CurrentColumn_Table=table(CurrentColumn)
    levels(CurrentColumn)[CurrentColumn_Table>Thres_Medium & CurrentColumn_Table<=Thres_High ]=paste(i,'_High',sep="")
    X[[i]]=CurrentColumn                    #Plug-back to the data.frame
  }
  #Finally, make sure we didn't miss any types
  if(!is.factor(CurrentColumn) & !is.numeric(CurrentColumn) & ExcludeVars[j]!=CurrentColumnVariableName){print(CurrentColumnName)}
  if(add){j=j+1}
}
#Get rid of those columns
print(ExcludeVars) #161 columns excluded
idx=!(names(X) %in% ExcludeVars);
XS=X[,!(names(X) %in% ExcludeVars),with=FALSE]

#####################NOTE: NEED TO ADD SUMMARIES#################
set.seed(666)
sample.index1=sample(nrow(XS),nrow(XS)*.8)
dt.train=XS[sample.index1]
dt.test=XS[-sample.index1]
sample.index2=sample(nrow(dt.train),nrow(dt.train)*.8)
dt.validation=dt.train[-sample.index2]
dt.train=dt.train[sample.index2]
#Done with cleaning!!!!
#3
#I'm working with churn
Y=Y_churn
y.train=Y[sample.index1,]
y.test=Y[-sample.index1,]
y.validation=y.train[-sample.index2]
y.train=y.train[sample.index2]

#Is there any reason not to just use a lasso here?
#But, I'm going to do the random forest model
num_trees = 500

test_params <- list()
for(t in test_params)

  rf = randomForest(
    x=dt.train[1:10000],
    y=as.factor(y.train[1:10000]), 
    xtests=dt.validation,  
    ntree=num_trees, 
    mtry=15,         # try 22 = sqrt(477) features at each split
    importance=TRUE, 
    nodesize=20     # need this many observations in the leaf
)

rf

varImpPlot(rf)3

vars <-randomForest::importance(rf, type = 2)
as.matrix(vars[order(vars), ])