
##############
# Homework 1 #
##############

########
# Load Config Files
########

	options("width" = 250)
	options(scipen  = 999)
	options(digits  = 003)

	library(ggplot2); require(gridExtra); library(MASS); library(Matrix); library(parallel)
	library(kknn); library(boot); library(rpart); library(data.table)
	library(gamlr); library(BayesTree); library(xgboost); library(ranger)

	set.seed(666) # the devils seed

	# install.packages("xgboost", dependencies = TRUE, INSTALL_opts = c('--no-lock'))

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

	lossMR <- function(y, phat, thr = 0.5){
					if(is.factor(y)) y <- as.numeric(y) - 1
  					yhat <- ifelse(phat > thr, 1, 0)
  					return(1 - mean(yhat == y))
	}

########
# Question 1
########

	########
	# Feature Enginnering 
	########

	bike.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_train.csv")))
	bike.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_test.csv")))
	
	bike.train[  ,UID   := .I]; setkey(bike.train, UID) # create UID column
	bike.test [  ,UID   := .I]; setkey(bike.test, UID) # create UID column
	bike.test [  ,count := NA] # these are to be predicted

	data <- rbind(bike.train, bike.test)	

	data[ ,':='( # factor categorical vars
			year = naref(factor(year)),
			month = naref(factor(month)),
			# daylabel = naref(factor(daylabel)), # toggle for continuous
			# day = naref(factor(day)),
			# hour = naref(factor(hour)),
			season = naref(factor(season)),
			holiday = naref(factor(holiday)),
			workingday = naref(factor(workingday)),
			weather = naref(factor(weather))
	)]

	dummies  <- model.matrix( ~ year + month + day + hour + season + holiday + workingday + weather, data = data)[,-1]
	all.data <- cbind(data[ ,c("count", "temp", "atemp", "humidity", "windspeed", "daylabel"), with = FALSE], dummies)

	trainX <- all.data[!is.na(count), !('count'), with = FALSE]
	trainY <- all.data[!is.na(count), ('count'), with = FALSE]
	testX <- all.data[is.na(count), !('count'), with = FALSE]
	
	trainOOS.idx <- sample(1:NROW(trainX), NROW(trainX) * 0.10) # sample 1/3 of the training data in order to get trueOOS estimate
	trainX.fit <- trainX[-trainOOS.idx, ] #use these for true testing and evaluating
	trainY.fit <- trainY[-trainOOS.idx, ]
	trainX.OOS <- trainX[ trainOOS.idx, ]
	trainY.OOS <- trainY[ trainOOS.idx, ]

	testX <- Matrix(as.matrix(all.data[is.na(count), !('count'), with = FALSE]))
	testY <- NULL

	########
	# Build Models
	########

	# LASSO <- cv.gamlr(trainX.fit, trainY.fit, family = 'gaussian', verb = TRUE, nfold = 10)
	# yhat.LASSO <- predict(LASSO, trainX.OOS)

	# LASSO.insample.RMSE <- sqrt(min(LASSO$cvm)) # in-sample CV RMSE
	# LASSO.outsample.RMSE <- sqrt(mean((as.matrix(trainY.OOS) - yhat.LASSO) ^ 2)) # OOS RMSE

	# params <- list(gamma = 0.02, max_depth = 20, booster = "gbtree", objective = "reg:linear") # patrick parameters
	# params <- list(max_depth = 4, booster = "dart", objective = "reg:linear") # nathan parameters 
	params <- list(gamma = 0.08, max_depth = 8, booster = "gbtree", objective = "reg:linear") # 41.8

	# CV Test Results
	# @ patrick data = patrick params: 57.6 nathan params: 
	# @ nathan data = patrick params: 53.6 nathan params: 
	# @ patrick data = patrick params #2 41.8 ! best parameters // 39.5

	#left is patrick data, right is nathan data

	XGBST.cv <- xgb.cv(params = params, data = as.matrix(apply(trainX, 2, as.numeric)), 
						label = as.vector(unlist(trainY)),
                       nthread = detectCores() - 1, verbose = 1, nfold = 10, nrounds = 2500)

	# odd numeric, gets test MSE 

	XGBST <- xgboost(params = params, data = as.matrix(trainX), label = as.vector(unlist(trainY)),
					 nthread = detectCores() - 1, verbose = 1, nrounds = 2500)

	yhat.XGBST <- predict(XGBST, as.matrix(trainX.OOS))
	XGBST.outsample.RSME <- sqrt(mean((as.matrix(trainY.OOS) - yhat.XGBST) ^ 2)); print(XGBST.outsample.RSME) # OOS RMSE

	yhat <- predict(XGBST, as.matrix(testX))
	predictions <- data.frame(count = ifelse(yhat < 0, 0, yhat))
	write.csv(predictions, file = 'hw2-1-matare.csv', row.names = FALSE)

  	fboost <- gbm(count ~., data = cbind.data.frame(trainY.fit, trainX.fit), distribution = "gaussian", n.trees = 2500, interaction.depth = 20, shrinkage = 0.8)
  
  	yhat.GBM <- predict(fboost, newdata = trainX.OOS, n.trees = 2500)
  	XGBST.outsample.RSME <- sqrt(mean((as.matrix(trainY.OOS) - yhat.GBM) ^ 2)); print(XGBST.outsample.RSME) # OOS RMSE

	# OOS Test results
	# @ patrick data = patrik params: ___ nathan params: 
	# @ nathan data = patrik params: ___ nathan params: 

	# RF <- ranger(count ~., 	data = cbind.data.frame(trainY.fit, trainX.fit), probability = FALSE, 
	# 						classification = FALSE, num.trees = 10000, write.forest = TRUE, 
	# 						num.threads = detectCores() - 1, importance = 'impurity', verbose = TRUE
	# )

	# yhat.RF <- predict(RF, trainX.OOS)$predictions 
	# RF.insample.RMSE <- sqrt(RF$prediction.error) # in-sample (OOB) RMSE
	# RF.outsample.RMSE <- sqrt(mean((as.matrix(trainY.OOS) - yhat.RF) ^ 2)) # OOS RMSE

	# BART <- bart(	x.train = as.data.frame(trainX.fit), 
	# 				y.train = as.double(unlist(trainY.fit)), 
	# 				x.test = as.data.frame(trainX.OOS),
	# 				ntree = 500, ndpost = 500, nskip = 250, verbose = TRUE, printevery = 10
	# 			)

	# 70.7 BART
	# sqrt(mean((trainY - BART$yhat.test.mean) ^ 2)) # BART$sigest

########
# Question 2
########

	########
	# Feature Enginnering 
	########

	movies.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/MovieReview_train.csv")))
	movies.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/MovieReview_test.csv")))
	
	movies.test[,sentiment := NA] # these are to be predicted
	all.data <- rbind(movies.train, movies.test)
	all.data <- as.data.table(apply(all.data, 2, as.double))

	binCols <- function(target, bins, data = all.data){
					binned.col <- cut(as.matrix(data[ ,target, with = FALSE]), bins, include.lowest = TRUE, labels = paste(target, ".bin.", 1:bins, sep = ""))
					return(binned.col)
	}

	all.data <- cbind(all.data, lengthAsbin = binCols(target = 'length', bins = 10))
	all.data[ ,length := NULL]

	trainX <- all.data[!is.na(sentiment), !('sentiment'), with = FALSE]
	trainY <- all.data[!is.na(sentiment), ('sentiment'),  with = FALSE]

	trainOOS.idx <- sample(1:NROW(trainX), NROW(trainX) * 0.20) # sample 1/3 of the training data in order to get trueOOS estimate
	trainX.fit <- trainX[-trainOOS.idx, ] #use these for true testing and evaluating
	trainY.fit <- trainY[-trainOOS.idx, ]
	trainX.OOS <- trainX[trainOOS.idx, ]
	trainY.OOS <- trainY[trainOOS.idx, ]

	toDouble <- function(data){ # turns data into double
					ind.num <- names(which(sapply(data, is.numeric)))
					ind.factor <- names(which(!sapply(data, is.numeric)))

					data.double <- apply(data[, ind.num, with = FALSE], 2, as.double)
					dummies <- model.matrix(~ lengthAsbin, naref(data[, ind.factor, with = FALSE]))[ ,-1] # currently not adaptive

					data.out <- cbind(data.double, dummies)
					return(data.out)
	}

	trainX <- toDouble(trainX)
	trainY  <- apply(trainY , 2, as.double)

	trainX.fit <- toDouble(trainX.fit)
	trainY.fit  <- apply(trainY.fit , 2, as.double)
	trainX.OOS <- toDouble(trainX.OOS)


	testX <- all.data[is.na(sentiment), !('sentiment'), with = FALSE]
	testX <- toDouble(testX)
	testY <- NULL

	########
	# Build Models 
	########

	LASSO <- cv.gamlr(trainX.fit, trainY.fit, family = 'binomial', gamma = 0, verb = TRUE, nfold = 10)
	prob.LASSO <- predict(LASSO, trainX.OOS, type = 'response') # 20.1 % w/ binning length 15.8% - 21% miss classification; depends on randomness from CV, and random test sample
	loss.LASSO <- lossMR(trainY.OOS, prob.LASSO); print(loss.LASSO)

	RF <- ranger(sentiment ~., 	data = cbind.data.frame(sentiment = factor(trainY.fit), trainX.fit), probability = TRUE, classification = TRUE, 
								num.trees = 10000, write.forest = TRUE, num.threads = detectCores() - 1, verbose = TRUE)

	prob.RF <- ranger:::predict.ranger(RF, trainX.OOS, type = 'response')$predictions[,'1']
	loss.RF <- lossMR(trainY.OOS, prob.RF); print(loss.RF) # 19.6 %

	LASSO <- cv.gamlr(trainX, trainY, family = 'binomial', gamma = 0, verb = TRUE, nfold = 10)
	phat <- predict(LASSO, as.matrix(testX), type = 'response')
	yhat <- ifelse(phat > 0.5, 1, 0)
	predictions <- data.frame(sentiment = yhat)

	write.csv(predictions, file = 'hw2-2-matare.csv', row.names = FALSE)

	# BART <- bart(x.train = as.data.frame(trainX.fit), y.train = as.double(unlist(trainY.fit)), 
	# 			 x.test = as.data.frame(trainX.OOS), ntree = 100, ndpost = 500, nskip = 250, verbose = TRUE, printevery = 10)

	# params <- list(max_depth = 4, nrounds = 2500, booster = "gbtree", objective = "binary:logistic") # 54.2
	# XGBST.cv <- xgb.cv(params = params, data = as.matrix(trainX.fit), label = as.vector(unlist(trainY.fit)),
 #                       nthread = detectCores() - 1, verbose = 0, nfold = 5)

	# # load best parameters into xgboost
	# XGBST <- xgboost(params = params, data = as.matrix(trainX.fit), label = as.vector(unlist(trainY.fit)),
	# 				 nthread = detectCores() - 1, verbose = 1, nrounds = 500)

	# yhat.XGBST <- predict(XGBST, as.matrix(trainX.OOS), type = 'prob')
	# loss.XGBST <- lossMR(trainY.OOS, yhat.XGBST)