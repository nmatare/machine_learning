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
	library(kknn); library(boot); library(rpart); library(data.table); library(foreach)
	library(doMC); library(doRNG)
	library(gamlr); library(BayesTree); library(xgboost); library(ranger)

	set.seed(666) # the devils seed

	username <- Sys.info()[["user"]]
	dir <- paste("/home/", username, "/projects/machine_learning/midterm/", sep = ""); setwd(dir)
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

	multiplot <- function(..., plotlist = NULL, file, cols=1, layout = NULL) {
	  require(grid)

	  # Make a list from the ... arguments and plotlist
	  plots <- c(list(...), plotlist)

	  numPlots = length(plots)

	  # If layout is NULL, then use 'cols' to determine layout
	  if (is.null(layout)) {
	    # Make the panel
	    # ncol: Number of columns of plots
	    # nrow: Number of rows needed, calculated from # of cols
	    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
	                    ncol = cols, nrow = ceiling(numPlots/cols))
	  }

	 if (numPlots==1) {
	    print(plots[[1]])

	  } else {
	    # Set up the page
	    grid.newpage()
	    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

	    # Make each plot, in the correct location
	    for (i in 1:numPlots) {
	      # Get the i,j matrix positions of the regions that contain this subplot
	      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

	      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
	                                      layout.pos.col = matchidx$col))
	    }
	  }# for plotting
	}

########
# Question 1
########


########
# Question 2
########
		
		data <- as.data.table(read.csv("PhillyCrime.csv"))

		############
		#  Part 1  #
		############

		ggplot(data = data, aes(x = X, y = Y, color = Category)) + geom_point() +
			scale_color_manual(values = c("salmon", "grey"), name = "Category", labels = c("Vandalism", "Thefts")) 

		############
		#  Part 2  #
		############

			data <- data[ ,.(Category, X, Y)]
			data[ , ':=' (
				Category = as.factor(Category),
				X 		 = as.numeric(X),
				Y 		 = as.numeric(Y)
			)]

			idx 	 <- sample(1:NROW(data), NROW(data) * 0.50) # 50% random split
			train 	 <- data[ idx, ]
			validate <- data[-idx, ]
		
			knnMR <- function(K){ # does KNN for selected K and reports missclassification rate
							knn 	<- kknn(formula = Category ~ X + Y, train = train, test = validate, kernel = "rectangular", k = K)
							yhat 	<- as.numeric(knn$fitted.values) - 1 # turn into binary; Vandalism = 1, Theft = 0
							true_y  <- as.numeric(validate$Category) - 1
							MR 		<- lossMR(true_y, yhat)
							return(MR)
			}

			result <- as.vector(NULL)
			for(k in 1:100) result[k] <- knnMR(k)

			#######
			# (A) #
			#######

			plotMissClass <- function(x){

						p 	<-	ggplot(data = data.frame(x), aes(x = 1:NROW(x), y = x)) + 
									xlab("K") + ylab("Misclassification Rate") +
									geom_line(color = 'grey') + 
									geom_point()
						return(p)
			}

			plotMissClass(result)

			########
			# (B)
			########

			which.min(result) # optimal k

			########
			# (C)
			########

			plotBestKNN <- function(x, train, validate){

							knn 	<- kknn(formula = Category ~ X + Y, train = train, test = validate, kernel = "rectangular", k = which.min(x)) # at best K
							DT 		<- cbind(validate, pred_Category = knn$fitted.values) # add the predicted column

							p <- ggplot(data = DT, aes(x = X, y = Y, color = pred_Category)) + 
										geom_point() +
										scale_color_manual(values = c("salmon", "grey"), name = "Predicted Category", labels = c("Vandalism", "Thefts"))
							return(p)				
			}

			plotBestKNN(x = result, train, validate)

		############
		#  Part 3  #
		############

			registerDoMC(detectCores() - 1) # detect number of cores to split work apart
			detectCores() # boothGrid is awesome @ 64 cores!

			idxs 	 <- replicate(20, sample(1:NROW(data), NROW(data) * 0.50)) # 20, 50% random split; must create outside of foreach
 			results  <- foreach(i = 1:20) %dopar% { # resample data 20 times and find optimal k on validation set using devils seed

 				idx 	 <- idxs[ ,i]					
				train 	 <- data[ idx, ]
				validate <- data[-idx, ]

				result <- as.vector(NULL)
				for(k in 1:100) result[k] <- knnMR(k)			
				return(list(result = result, train = train, validate = validate))
			}

			#######
			# (A) #
			#######

			plots <- list() 
			for(n in 1:20) plots[[n]] <- plotMissClass(results[[n]]$result)

			multiplot(plotlist = plots, cols = 5)

			#######
			# (B) #
			#######

			plots <- list()
			for(n in 1:10) plots[[n]] <- plotBestKNN(x = results[[n]]$result, results[[n]]$train, results[[n]]$validate)
			
			multiplot(plotlist = plots, cols = 5)

			plots <- list()
			for(n in 11:20) plots[[n]] <- plotBestKNN(x = results[[n]]$result, results[[n]]$train, results[[n]]$validate)

			multiplot(plotlist = plots, cols = 5)	

			#######
			# (C) #
			#######

			t(matrix(lapply(results, function(x) which.min(x$result)), dimnames = list(1:length(results), "K"))) # best K at each N
	
			#######
			# (D) #
			#######

			mean(unlist(lapply(results, function(x) min(x$result)))) # mean of minimumn OOS classification rate
			sd(unlist(lapply(results, function(x) min(x$result))))

		############
		#  Part 4  #
		############

			idxs 	 <- replicate(20, sample(1:NROW(data), NROW(data) * 0.90)) # 20, 90% random split; must create outside of foreach
 			results  <- foreach(i = 1:20) %dopar% { # resample data 20 times and find optimal k on validation set using devils seed

 				idx 	 <- idxs[ ,i]					
				train 	 <- data[ idx, ]
				validate <- data[-idx, ]

				result <- as.vector(NULL)
				for(k in 1:100) result[k] <- knnMR(k)			
				return(list(result = result, train = train, validate = validate))
			}

			#######
			# (A) #
			#######

			plots <- list() 
			for(n in 1:20) plots[[n]] <- plotMissClass(results[[n]]$result)

			multiplot(plotlist = plots, cols = 5)

			#######
			# (B) #
			#######

			plots <- list()
			for(n in 1:10) plots[[n]] <- plotBestKNN(x = results[[n]]$result, results[[n]]$train, results[[n]]$validate)
			
			multiplot(plotlist = plots, cols = 5)

			plots <- list()
			for(n in 11:20) plots[[n]] <- plotBestKNN(x = results[[n]]$result, results[[n]]$train, results[[n]]$validate)

			multiplot(plotlist = plots, cols = 5)	

			#######
			# (C) #
			#######

			t(matrix(lapply(results, function(x) which.min(x$result)), dimnames = list(1:length(results), "K"))) # best K at each N
	
			#######
			# (D) #
			#######

			mean(unlist(lapply(results, function(x) min(x$result)))) # average min OOS error
			sd(unlist(lapply(results, function(x) min(x$result))))

		############
		#  Part 5  #
		############

			# Comment on the difference between the results obtained in (2), (3) and (4).

		############
		#  Part 6  #
		############

		idx 	 <- sample(1:NROW(data), NROW(data) * 0.50) # 50% random split
		train 	 <- data[ idx, ]
		validate <- data[-idx, ]

		knn 	<- kknn(formula = Category ~ X + Y, train = train, test = validate, kernel = "rectangular", k = 25) # at best K
		yhat 	<- as.numeric(knn$fitted.values) - 1 # turn into binary; Vandalism = 1, Theft = 0
		true_y  <- as.numeric(validate$Category) - 1

		pROC:::roc(response = true_y, predictor = yhat, plot = TRUE) # ROC curve, AUC is 0.665

########
# Question 3
########