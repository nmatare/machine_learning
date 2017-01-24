
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
	# Feature Enginnering 
	########

	bike.train <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_train.csv")))
	bike.test <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/hw02/Bike_test.csv")))
	
	bike.train[,UID := .I]; setkey(bike.train, UID) # create UID column
	bike.test[,UID := .I]; setkey(bike.test, UID) # create UID column
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

	trainX <- Matrix(as.matrix(all.data[!is.na(count), !('count'), with = FALSE]))
	trainY <- all.data[!is.na(count),  ('count'), with = FALSE]

	testX <- Matrix(as.matrix(all.data[is.na(count), !('count'), with = FALSE]))
	testY <- NULL

	########
	# Build Models
	########

	## TO DO


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