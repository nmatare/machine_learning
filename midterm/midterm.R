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
					if(is.factor(y)) 
						y <- as.numeric(y) - 1
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
		
		# Part 1

		# A) FALSE :: lec 1 slide 32
		# B) TRUE :: lec 1 slide 32
		# C) TRUE :: Consider kNN at K = 1, the model will be perfectly fit to its nearest neighboor, itself; as K increases the model becomes more general and incorpates further neighbores; thus, misclassification increases
		# D) Depends :: Dependent on the K, and the nature of the dataset, as K increases the model could misclassify worse or better: insert lec 1 slide 29 picture

		# Part 2

		# A) TRUE :: lec 1 slide 32 :: more complex
		# B) FALSE :: lec 1 slide 32 :: less complex, reference gunter's class notes
		# C) TRUE :: it's overfit to the training data
		# D) FALSE :: it's underfit, model 2 would likely have the best the testing 

		# Part 3

		# Usually true, but it could depend on the data. The data used to evaluate the model based of the training data could by random chance present a better fit, this is why multiple iterations of cross validation needs
		# to be performed to make sure you're not observing a chance example; see lecture 1

		# Part 4
		# Slightly FALSE, leave one out cross validation is an approximate unbiased estimator of the true error; although this is close theoritical reasons for why it is not exactly completely unbiased
		# However, leave-one-cross validation disallows for much of the hetrogenitiy in the dataset, and can produce underestimated predictions; this and compounded with computation expenseness 5 or 10 fold CV is widely used


########
# Question 2
########
		
		data <- as.data.table(read.csv("PhillyCrime.csv"))

		############
		#  Part 1  #
		############

		ggplot(data = data, aes(x = X, y = Y, color = Category)) + 
			geom_point() +
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
		
			knnMR <- function(K, train, validate){ # does KNN for selected K and reports missclassification rate
							knn 	<- kknn(formula = Category ~ X + Y, train = train, 
											test = validate, kernel = "rectangular", k = K)
							yhat 	<- as.numeric(knn$fitted.values) - 1 # turn into binary; Vandalism = 1, Theft = 0
							true_y  <- as.numeric(validate$Category) - 1
							MR 		<- lossMR(true_y, yhat)
							return(MR)
			}

			result <- as.vector(NULL)
			for(k in 1:100) result[k] <- knnMR(k, train, validate)

			#######
			# (A) #
			#######

			ggplot(data = data.frame(result), aes(x = 1:NROW(result), y = result)) + 
					xlab("K") + ylab("Misclassification Rate") +
					geom_line(color = 'grey') + 
					geom_point()

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
				for(k in 1:100) result[k] <- knnMR(k, train, validate)
				return(list(result = result, train = train, validate = validate))
			}

			#######
			# (A) #
			#######

			missclass <- as.data.frame(cbind(1:100, do.call(cbind, lapply(results, function(x) x$result))))
			missclass_long <- melt(missclass, id = "V1")
	
			ggplot(data = missclass_long, aes(x = V1, y = value, group = variable)) +
				geom_line(color = 'grey') +
				stat_summary(fun.y = mean, geom = "line", lwd = 1, aes(group = 1)) +
				xlab("K") + ylab("Misclassification Rate") 

			#######
			# (B) #
			#######

			n <- 15
			plotBestKNN(x = results[[n]]$result, results[[n]]$train, results[[n]]$validate)

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
			min.mr <- unlist(lapply(results, function(x) min(x$result)))
			sd(min.mr) / sqrt(length(min.mr))

		############
		#  Part 4  #
		############

			idxs 	 <- replicate(20, sample(1:NROW(data), NROW(data) * 0.90)) # 20, 90% random split; must create outside of foreach
 			results  <- foreach(i = 1:20) %dopar% { # resample data 20 times and find optimal k on validation set using devils seed

 				idx 	 <- idxs[ ,i]					
				train 	 <- data[ idx, ]
				validate <- data[-idx, ]

				result <- as.vector(NULL)
				for(k in 1:100) result[k] <- knnMR(k, train, validate)
				return(list(result = result, train = train, validate = validate))
			}

			#######
			# (A) #
			#######

			missclass <- as.data.frame(cbind(1:100, do.call(cbind, lapply(results, function(x) x$result))))
			missclass_long <- melt(missclass, id = "V1")
	
			ggplot(data = missclass_long, aes(x = V1, y = value, group = variable)) +
				geom_line(color = 'grey') +
				stat_summary(fun.y = mean, geom = "line", lwd = 1, aes(group = 1)) +
				xlab("K") + ylab("Misclassification Rate") 

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
			min.mr <- unlist(lapply(results, function(x) min(x$result)))
			sd(min.mr) / sqrt(length(min.mr))

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

	# do weird shit
	matrix(2, 1)

	cost <- 15 # fixed cost of targeting
	benefit <- # benefit of responding
	phat <- # probabilty of responding

	phat * benefit + (1 - phat) * cost > 0 # solve for phat

	# determine probability of response via classification model
	# determine donation amount given response back

########
# Question 4
########

		############
		#  Part 1  #
		############

		data <- as.data.table(read.csv("Tayko.csv"))
		data[ , ':=' (

			US = as.logical(US),

			source_a 		 = as.logical(source_a),
			source_c 		 = as.logical(source_c),
			source_b 		 = as.logical(source_b),
			source_d 		 = as.logical(source_d),
			source_e 		 = as.logical(source_e),
			source_m 		 = as.logical(source_m),
			source_o 		 = as.logical(source_o),
			source_h 		 = as.logical(source_h),
			source_r 		 = as.logical(source_r),
			source_s 		 = as.logical(source_s),
			source_t 		 = as.logical(source_t),
			source_u 		 = as.logical(source_u),
			source_p 		 = as.logical(source_p),
			source_x 		 = as.logical(source_x),
			source_w 		 = as.logical(source_w),

			Freq 		 				 = as.numeric(Freq),
			last_update_days_ago 		 = as.numeric(last_update_days_ago),
			first_update_days_ago 		 = as.numeric(first_update_days_ago),
			Web_order 		 			 = as.logical(Web_order),
			Gender_is_male 		 		 = as.logical(Gender_is_male),
			Address_is_res 		 		 = as.logical(Address_is_res),
			Purchase 		 			 = as.logical(Purchase),
			Spending 		 			 = as.numeric(Spending),
			Partition 		 			 = as.factor(Partition)
		)]

		total.spend <- sum(data[Partition == 's', Spending])
		num.customer <- length(data[Partition == 's', Spending])

		(total.spend / num.customer * 0.107 - 2) * 18e4 / 1e6 # gross profit in millions

		############
		#  Part 2  #
		############

		train_p <- data[Partition %in% c('t', 'v'), !which(colnames(data) %in% c('Spending', 'sequence_number', 'Partition')), with = FALSE] # combine training and validation together; remove spending
		test_p <- data[Partition %in% c('s'), !which(colnames(data) %in% c('Spending', 'sequence_number', 'Partition')), with = FALSE] 

		# Try Linear
		LASSO <- cv.gamlr(	x = train_p[ ,which(colnames(train_p) != 'Purchase'), with = FALSE], 
							y = train_p[ ,Purchase], 
							family = 'binomial', verb = FALSE, lambda.start = 0.1, nfold = 10)

		yhat <- predict(LASSO, newdata = train_p[ ,which(colnames(train_p) != 'Purchase'), with = FALSE], type = 'response')
		
		lossMR(train_p[ ,Purchase], yhat) # LASSO prediction error

		# Try RF
		RF <- ranger(	as.factor(Purchase) ~., 	data = train_p, 
						probability = TRUE, classification = TRUE, num.trees = 5000, write.forest = TRUE, 
						num.threads = detectCores() - 1, importance = 'impurity', verbose = TRUE
				)

		RF$prediction.error * 100 # OOB prediction error

		# Try Boosting
		params <- list(gamma = 0.08, max_depth = 4, booster = "gbtree", objective = "binary:logistic") 
		XGBST.cv <- xgb.cv(	params = params, 
							data = as.matrix(train_p[ ,which(colnames(train_p) != 'Purchase'), with = FALSE]), 
							label = as.vector(train_p[ ,Purchase]),
                       		nthread = detectCores() - 1, verbose = 1, nfold = 10, nrounds = 100)

		tail(XGBST.cv$evaluation$test_error_mean, 1) # last cv.fold missclassification error

		############
		#  Part 3  #
		############

		phat <- ranger:::predict.ranger(RF, test_p[ ,which(colnames(test_p) != 'Purchase'), with = FALSE])$predictions[ ,'TRUE']	
		pROC:::roc(response = test_p[ ,Purchase], predictor = phat, plot = TRUE) # ROC curve, AUC is 0.923

		############
		#  Part 4  #
		############

		train_s <- data[Partition %in% c('t', 'v') & Purchase == TRUE, !which(colnames(data) %in% c('sequence_number', 'Partition', 'Purchase')), with = FALSE] # combine training and validation together; remove spending
		test_s <- data[Partition %in% c('s'), !which(colnames(data) %in% c('sequence_number', 'Partition', 'Purchase')), with = FALSE] 

		# Try Linear
		LASSO <- cv.gamlr(	x = train_s[ ,which(colnames(train_s) != 'Spending'), with = FALSE], 
							y = train_s[ ,Spending], 
							family = 'gaussian', verb = FALSE, lambda.start = Inf, nfold = 10)

		sqrt(LASSO$cvm[which.min(LASSO$cvm)]) # RMSE

		# Try RF
		RF <- ranger(	Spending ~., 	data = train_s, 
						probability = FALSE, classification = FALSE, num.trees = 5000, write.forest = TRUE, 
						num.threads = detectCores() - 1, importance = 'impurity', verbose = TRUE
				)

		sqrt(RF$prediction.error) # RMSE

		# Try Boosting
		params <- list(gamma = 0.08, max_depth = 4, booster = "gbtree", objective = "reg:linear") 
		XGBST.cv <- xgb.cv(	params = params, 
							data = as.matrix(train_s[ ,which(colnames(train_s) != 'Spending'), with = FALSE]), 
							label = as.vector(train_s[ ,Spending]),
                       		nthread = detectCores() - 1, verbose = 1, nfold = 10, nrounds = 100)

		tail(XGBST.cv$evaluation$test_rmse_mean, 1) # last cv.fold missclassification error; best RMSE

		############
		#  Part 5  #
		############

		XGBST <- xgboost(	params = params, data = as.matrix(train_s[ ,which(colnames(train_s) != 'Spending'), with = FALSE]), 
							label = as.vector(train_s[ ,Spending]), nthread = detectCores() - 1, verbose = 1, nrounds = 500)

		yhat <- predict(XGBST, as.matrix(test_s[ ,which(colnames(test_s) != 'Spending'), with = FALSE])) # predicted spending given best model
		yhat <- ifelse(yhat <= 0, 0, yhat) # not possible for neg yhats

		test_p[ ,phat := phat] # phats to purchase data.table
		test_p[ ,yhat := yhat] # yhats to those that purchased
		total.spend <- sum(data[Partition == 's' ,Spending])
		num.customer <- length(data[Partition == 's' ,Spending])

		expected_spending <- test_p[ ,cumsum(sort(yhat * phat * 0.107)) / (total.spend / num.customer)]
		ggplot(data = data.frame(expected_spending), aes(x = 1:NROW(expected_spending), y = expected_spending)) + geom_line() #xlab("K") + ylab("Misclassification Rate") 

		############
		#  Part 6  #
		############

		mr <- (total.spend / num.customer * 0.107 - 2) # marginal revenue from Q4.1
		n <- 500 * 18e4 / 5e6 # mail the top 3.6% of the list 
		lift <- test_p[ ,cumsum(sort(phat * yhat))][n] # lift at n percentile
		(lift * (total.spend / num.customer * 0.107) - 2) * 18e4 / 1e6 # new gross profit

