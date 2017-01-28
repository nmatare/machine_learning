
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
	# Feature Enginnering 
	########

	bike.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_train.csv")))
	bike.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_test.csv")))
	
	bike.train[ ,UID := .I]; setkey(bike.train, UID) # create UID column
	bike.test[ ,UID := .I]; setkey(bike.test, UID) # create UID column
	bike.test[,count := NA] # these are to be predicted

	data <- rbind(bike.train, bike.test)	

	data[ ,':='( # factor categorical vars
			year = naref(factor(year)),
			month = naref(factor(month)),
			daylabel = naref(factor(daylabel)),
			day = naref(factor(day)),
			hour = naref(factor(hour)),
			season = naref(factor(season)),
			holiday = naref(factor(holiday)),
			workingday = naref(factor(workingday)),
			weather = naref(factor(weather))
	)]

	dummies <- model.matrix( ~ year + month + day + hour + season + holiday + workingday + weather + daylabel, data = data)[,-1]
	all.data <- cbind(data[ ,c("count", "temp", "atemp", "humidity", "windspeed"), with = FALSE], dummies)

	trainX <- all.data[!is.na(count), !('count'), with = FALSE]
	trainY <- all.data[!is.na(count), ('count'), with = FALSE]
	
	trainOOS.idx <- sample(1:NROW(trainX), NROW(trainX) * 0.33) # sample 1/3 of the training data in order to get trueOOS estimate
	trainX.fit <- trainX[-trainOOS.idx] #use these for true testing and evaluating
	trainY.fit <- trainY[-trainOOS.idx]
	trainX.OOS <- trainX[trainOOS.idx]
	trainY.OOS <- trainY[trainOOS.idx]

	testX <- Matrix(as.matrix(all.data[is.na(count), !('count'), with = FALSE]))
	testY <- NULL

	########
	# Build Models
	########

	LASSO <- cv.gamlr(trainX.fit, trainY.fit, family = 'gaussian', verb = TRUE, nfold = 10)
	yhat.LASSO <- predict(LASSO, trainX.OOS)

	LASSO.insample.RMSE <- sqrt(min(LASSO$cvm)) # in-sample CV RMSE
	LASSO.outsample.RMSE <- sqrt(mean((as.matrix(trainY.OOS) - yhat.LASSO) ^ 2)) # OOS RMSE

	depths <- c(2, 3, 4, 5, 6); results <- list()
	for(depth in depths){

		XGBST <- xgboost(	data = as.matrix(trainX.fit), label = as.vector(unlist(trainY.fit)), max_depth = depth, 
							nthread = detectCores() - 1, nrounds = 10000, verbose = 0, objective = "reg:linear")

		yhat.XGBST <- predict(XGBST, as.matrix(trainX.OOS))
		XGBST.outsample.RSME <- sqrt(mean((as.matrix(trainY.OOS) - yhat.XGBST) ^ 2)) # OOS RMSE
		results[[depth]] <- XGBST.outsample.RSME
		print(paste("Finished depth:", depth, "OOS MSE:", round(XGBST.outsample.RSME)))
	}
		# best model is found at RMSE 50 at nrounds 10000 depth 5


	RF <- ranger(count ~., 	data = cbind.data.frame(trainY.fit, trainX.fit), probability = FALSE, 
							classification = FALSE, num.trees = 10000, write.forest = TRUE, 
							num.threads = detectCores() - 1, importance = 'impurity', verbose = TRUE
				)

	yhat.RF <- predict(RF, trainX.OOS)$predictions 
	RF.insample.RMSE <- sqrt(RF$prediction.error) # in-sample (OOB) RMSE
	RF.outsample.RMSE <- sqrt(mean((as.matrix(trainY.OOS) - yhat.RF) ^ 2)) # OOS RMSE

	BART <- bart(	x.train = as.data.frame(trainX.fit), 
					y.train = as.double(unlist(trainY.fit)), 
					x.test = as.data.frame(trainX.OOS),
					ntree = 500, ndpost = 500, nskip = 250, verbose = TRUE, printevery = 10
				)

	# 70.7 BART
	sqrt(mean((trainY - BART$yhat.test.mean) ^ 2)) # BART$sigest

########
# Question 2
########

	########
	# Feature Enginnering 
	########

	movies.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/MovieReview_train.csv")))
	movies.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/MovieReview_test.csv")))
	
	movies.train[,UID := .I]; setkey(movies.train, UID) # create UID column
	movies.test[,UID := .I]; setkey(movies.test, UID) # create UID column
	movies.test[,sentiment := NA] # these are to be predicted

	all.data <- rbind(movies.train, movies.test)	

	trainX <- Matrix(as.matrix(all.data[!is.na(sentiment), !('sentiment'), with = FALSE]))
	trainY <- all.data[!is.na(sentiment), ('sentiment'), with = FALSE]

	testX <- Matrix(as.matrix(all.data[is.na(sentiment), !('sentiment'), with = FALSE]))
	testY <- NULL

	########
	# Build Models 
	########

	## TO DO