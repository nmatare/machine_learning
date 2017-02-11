##############
# Midterm
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

	dir <- "~/Documents/Education/Chicago_Booth/Classes/41204_Machine_Learning/machine_learning/midterm"
	setwd(dir)

	# Helper functions from TA
	mse <- function(y,yhat) {return(sum((y - yhat) ^ 2))}

	doknn <- function(x, y, xp, k){
					kdo=k[1]
					train = data.frame(x,y=y)
					test = data.frame(xp); names(test) = names(train)[1:(ncol(train)-1)]
					near  = kknn(y~.,train,test,k=kdo,kernel='rectangular')
					return(near$fitted)
	}

	docv <- function(x, y, set, predfun, loss, nfold = 10, doran = TRUE, verbose = TRUE, ...){
					#a little error checking
					if(!(is.matrix(x) | is.data.frame(x))) {cat('error in docv: x is not a matrix or data frame\n'); return(0)}
					if(!(is.vector(y))) {cat('error in docv: y is not a vector\n'); return(0)}
					if(!(length(y)==nrow(x))) {cat('error in docv: length(y) != nrow(x)\n'); return(0)}

					nset = nrow(set); n=length(y) #get dimensions
					if(n==nfold) doran=FALSE #no need to shuffle if you are doing them all.
					cat('in docv: nset,n,nfold: ',nset,n,nfold,'\n')
					lossv = rep(0,nset) #return values
					if(doran) {ii = sample(1:n,n); y=y[ii]; x=x[ii,,drop=FALSE]} #shuffle rows

					fs = round(n/nfold) # fold size
					for(i in 1:nfold) { #fold loop
					bot=(i-1)*fs+1; top=ifelse(i==nfold,n,i*fs); ii =bot:top
					if(verbose) cat('on fold: ',i,', range: ',bot,':',top,'\n')
					xin = x[-ii,,drop=FALSE]; yin=y[-ii]; xout=x[ii,,drop=FALSE]; yout=y[ii]
						for(k in 1:nset) { #setting loop
						  yhat = predfun(xin,yin,xout,set[k,],...)
						  lossv[k]=lossv[k]+loss(yout,yhat)
						} 
					} 
	  				return(lossv)
	}

	docvknn <- function(x, y, k, nfold = 10, doran = TRUE, verbose = TRUE){return(docv(x, y, matrix(k, ncol = 1), doknn, mse, nfold = nfold, doran = doran, verbose = verbose))}

	lossMR <- function(y, phat, thr = 0.5){
					if(is.factor(y)) y = as.numeric(y) - 1
					yhat <- ifelse(phat > thr, 1, 0)
					return(1 - mean(yhat == y))
	}

########
# Question 1
########


########
# Question 2
########
		
		data <- as.data.table(read.csv("PhillyCrime.csv"))

		########
		# (A)
		########

		ggplot(data = data, aes(x = X, y = Y, color = Category)) + geom_point() +
			scale_color_manual(values = c("salmon", "grey"), name = "Category", labels = c("Vandalism", "Thefts")) 

		########
		# (B)
		########

		data <- data[ ,.(Category, X, Y)]
		data[ , ':=' (
			Category =  as.numeric(Category) - 1, # turn into binary; Vandalism = 1, Theft = 0
			X = as.numeric(X),
			Y = as.numeric(Y)
		)]

		idx 	 <- sample(1:NROW(data), NROW(data) * 0.50) # 50% random split
		train 	 <- data[ idx, ]
		validate <- data[-idx, ]

		knnMR <- function(K){ # does KNN for selected K and reports missclassification rate
						knn 	<- kknn(formula = Category ~ X + Y, train = train, test = validate, kernel = "rectangular", k = K)
						yhat 	<- knn$fitted.values
						true_y  <- validate$Category
						MR 		<- lossMR(true_y, yhat)
						return(MR)
		}

		result <- as.vector(NULL)
		for(k in 1:100) result[k] <- knnMR(k)

		which.min(result) # optimal k

		########
		# (C)
		########

		results <- list()
		for(n in 1:20){ # resample data 20 times and find optimal k on validation set

			idx 	 <- sample(1:NROW(data), NROW(data) * 0.50) # 50% random split
			train 	 <- data[ idx, ]
			validate <- data[-idx, ]

			result <- as.vector(NULL)
			for(k in 1:100) result[k] <- knnMR(k)
			
			results[[n]] <- result # store n iteration into list
		}

