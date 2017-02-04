# This is a corollary example to the homework.

# In linear regression, we commonly transform categorical variables into dummy representation; this allows us to control for effects, say day or month. However, when playing around with boosted trees, it appears that we get better predictions when categorical features are represented as discrete features rather than when they are encoded as dummy variables. 

# A dummy example below:

# load data #
set.seed(90)
library(gbm)
library(randomForest)
data(airquality)
data <- airquality[complete.cases(airquality), ]

## as discrete ##

gboost1 <- gbm(	Ozone ~., data = data, distribution = "gaussian", 
		cv.folds = 10, n.trees = 10000, interaction.depth = 4, shrinkage = 0.8)
gboost1$cv.error[which.min(gboost1$cv.error)]
# [1] 439.14

## as factor ##

data$Month <- factor(data$Month)
data$Day <- factor(data$Day)

gboost2 <- gbm(	Ozone ~., data = data, distribution = "gaussian", 
		cv.folds = 10, n.trees = 10000, interaction.depth = 4, shrinkage = 0.8)
gboost2$cv.error[which.min(gboost2$cv.error)]
# [1] 770.1819


## as dummy ##

data$Month <- factor(data$Month, levels=c(NA, levels(data$Month)),exclude=NULL) # make reference level NA
data$Day <- factor(data$Day, levels=c(NA, levels(data$Day)),exclude=NULL)
mm <- model.matrix(~ Month + Day, data = data)[ ,-1]
data <- cbind(data[ ,-(5:6)], mm) # remove old factors and replace with dummies

gboost3 <- gbm(	Ozone ~., data = data, distribution = "gaussian", 
		cv.folds = 10, n.trees = 10000, interaction.depth = 4, shrinkage = 0.8)
gboost3$cv.error[which.min(gboost3$cv.error)]
# [1] 452.0293

