library(gbm)
library(randomForest)
set.seed(666)

data(airquality)
data <- airquality[complete.cases(airquality),]

# params <- list(max_depth = 8, booster = "gbtree", objective = "reg:linear") # 41.8
# data$Month <- factor(data$Month)
# data$Day <- factor(data$Day)

gboost1 <- gbm(	Ozone ~., data = data, distribution = "gaussian", 
				cv.folds = 10, n.trees = 5000, interaction.depth = 4, shrinkage = 0.8)
head(gboost1$cv.error, 1) 

# XGBST1.cv <- xgb.cv(	params = params, data = as.matrix(apply(data[,-1], 2, as.numeric)), 
# 						label = as.matrix(data[,1]), nthread = detectCores() - 1, verbose = 1, 
# 						nfold = 10, nrounds = 500)
# tail(XGBST1.cv$evaluation_log$test_rmse_mean,1)


############ 
data$Month <- factor(data$Month)
data$Day <- factor(data$Day)
data$Month <- factor(data$Month, levels=c(NA, levels(data$Month)),exclude=NULL) # make reference level NA
data$Day <- factor(data$Day, levels=c(NA, levels(data$Day)),exclude=NULL)
mm <- model.matrix(~ Month + Day, data = data)[ ,-1]
data <- cbind(data[ ,-(5:6)], mm) # remove old factors and replace with dummies

gboost2 <- gbm(	Ozone ~., data = data, distribution = "gaussian", 
				cv.folds = 10, n.trees = 5000, interaction.depth = 4, shrinkage = 0.8)
head(gboost2$cv.error, 1)


# XGBST2.cv <- xgb.cv(	params = params, data = as.matrix(apply(data[,-1], 2, as.numeric)), 
# 						label = as.matrix(data[,1]), nthread = detectCores() - 1, verbose = 1, 
# 						nfold = 10, nrounds = 500)

# tail(XGBST2.cv$evaluation_log$test_rmse_mean, 1)