
##############
# Homework 1 #
##############

########
# Load Config Files
########

	options("width" = 250)
	options(scipen = 999)
	options(digits = 3)

	library(ggplot2); require(gridExtra); library(MASS); library(kknn)
	library(boot); library(rpart); library(data.table)

	UC <- as.data.table(read.csv(url("https://raw.githubusercontent.com/ChicagoBoothML/DATA___UsedCars/master/UsedCars.csv")))
	UC[,UID := .I];	setkey(UC, UID) # create UID column
	set.seed(666) # the devils seed

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

########
# Question 1
########

	makeSimulation <- function(equation, num.train = 100, noise = 0){
					# noise controls the number of superfurlous predictor vars added to dataset

					# create training dataset
					x <- rnorm(num.train, mean = 0, sd = 1) #predictor x
					x.noise <- NULL					
					if(noise > 0) x.noise <- as.data.frame(sapply(rep(num.train, noise), function(w) rnorm(w, mean = 0, sd = 1))) # generate noise variables from normal distribution

					e <- rnorm(num.train, mean = 0, sd = 1) # random noise
					y <- eval(equation) + e # y is the function plus some noise

					train <- as.data.frame(cbind(y, x, x.noise)) #train models on predictor and noise
		
					test <- list() # create testing dataset
					for(i in 1:100){ #do 100 times to get 10,000 observations

						x <- rnorm(100, mean = 0, sd = 1)
						e <- rnorm(100, mean = 0, sd = 1) # random noise
						y <- eval(equation) + e

						x.noise <- NULL					
						if(noise > 0) x.noise <- as.data.frame(sapply(rep(100, noise), function(w) rnorm(w, mean = 0, sd = 1))) # generate noise variables from normal distribution

						test[[i]] <- as.data.frame(cbind(y, x, x.noise))  # true model has no noise so don't include it in the test set
					}

					test <- do.call(rbind.data.frame, test)
					stopifnot(isTRUE(dim(test)[1] == 10000)) # sanity check
					return(list(train = train, test = test))
	}

	makeKnn <- function(data, K){ # wrapper function for Knn
					knn <- kknn(		formula = y ~ ., # where . is everything or x or x1, x2, x3, etc
										train = data$train, # train on relationship between x and y
										test = data$test, # given x, find y 
										kernel = "rectangular", 
										k = K)
					return(knn$fitted.values)
	}

	doQuestionOne <- function(equation, Ks = 2:15, ...){

					#p1/p3 control verbosity of print
					simulation <- makeSimulation(equation, ... = ...)

					# Part 2 and 3
					base.plot <- ggplot(	data = simulation$test, aes(x = x, y = y)) + 
											geom_point(color = "darkgrey", size = 1, alpha = 3 / 5) + # show the relationship
											stat_function(fun = function(x) eval(equation), colour = "black") + # show the true linear equation
											geom_text(aes(x = max(x) - 1, y = 8, label = 'True Function', sep = ""), vjust = -4, size = 4, color = "black") # add text

					plot1 <- 	base.plot + ggtitle("Linear Model")	+		
								geom_smooth(method = "lm", col = 'blue', linetype = "dashed", show.legend = TRUE) + # show the true line; this is essentially lm(y ~ x, data = simulation)
								geom_text(aes(x = max(x) - 1, y = 8, label = 'Linear Fit', sep = ""), vjust = 4, size = 4, color = "blue")

					# Part 4
					knn2fitted <- 	cbind.data.frame(fit = makeKnn(K = 2, data = simulation), x = simulation$test$x) #fit on training data and then output the fitted y values and corresponding x's
					knn12fitted <- 	cbind.data.frame(fit = makeKnn(K = 12, data = simulation), x = simulation$test$x)

					plot.knn2 <-  base.plot + geom_line(data = knn2fitted, aes(y = fit, x = x), col = "green") + ggtitle("KNN at K = 2")
					plot.knn12 <- base.plot + geom_line(data = knn12fitted, aes(y = fit, x = x), col = "blue") + ggtitle("KNN at K = 12")

					plot2 <- arrangeGrob(plot.knn2, plot.knn12, ncol = 2)

					# Part 5
					MSEs <- list()
					for(k in Ks){
						fitted <- makeKnn(K = k, data = simulation) # get fitted vals
						MSEs[k] <- mean((simulation$test$y - fitted) ^ 2) # get MSE
					}

					MSEs <- cbind.data.frame(K = Ks, MSE = do.call(rbind, MSEs)) # get MSEs for each K (K here is really -log(1/K))
					linear <- lm(y ~ x, data = simulation$train) # fit linear model
					yhat <- predict(linear, simulation$test)
					MSE.lin <- mean((simulation$test$y - yhat) ^ 2)

					plot3 <- ggplot(		data = MSEs, aes(x = K, y = MSE)) + geom_point() + # plot MSE of all Ks
											geom_hline(yintercept = MSE.lin, col = "blue", linetype = "dashed") + # linear MSE
											xlab("log(1/K)") +
											ggtitle("MSEs of KNNs")
	

					return(list(plot1 = plot1, plot2 = plot2, plot3 = plot3))								
	}

	########
	# Simulation 1
	########

	question1 <- doQuestionOne(equation = quote(2 + 1.8 * x), noise = 0)

	question1$plot1; dev.new()
	grid.arrange(question1$plot2); dev.new()
	question1$plot3

	########
	# Simulation 2
	########

	question2 <- doQuestionOne(equation = quote(3 + exp(x + 1)), noise = 0)

	question2$plot1; dev.new()
	grid.arrange(question2$plot2); dev.new()
	question2$plot3

	########
	# Simulation 3
	########

	question3 <- doQuestionOne(equation = quote(2 + sin(2 * x)), noise = 0)
	
	question3$plot1; dev.new()
	grid.arrange(question3$plot2); dev.new()
	question3$plot3

	########
	# Simulation 4
	########

	plots <- list() 
	for(n in 1:20) plots[[n]] <- doQuestionOne(noise = n, equation = quote(2 + sin(2 * x))) # incrementally add noise
	
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

	multiplot(plotlist = lapply(plots, function(x) x$plot3), cols = 5)

	# Explanation: 
	# The superfluous features have no predictive power. Thus, when the amount of noise increases in the dataset, the Knn algorithim uses spurious features to predict the value of y.
	# If K is small and the number of superfluous features is large, then there is a high likeihood that y-hat is predicted by noise. 
	# As K increases, then the algorithim uses more features to predict y-hat; thus the likeihood that the features are spurious decreases, giving lower MSE. 
	# This, however, is notwithstanding the generated noise that greatly deteoriates the predictive power of the Knn algoritm as noted by the 
	# continued average decrease in Knn MSE relative to the MSE of the linear model.

	########
	# BONUS
	########

	q1 <- doQuestionOne(num.train = 100, noise = 5, equation = quote(2 + sin(2 * x))) # given training dataset with 100
	grid.arrange(q1$plot2); dev.new()

	q2 <- doQuestionOne(num.train = 1000, noise = 5, equation = quote(2 + sin(2 * x))) # given training dataset with 1000
	grid.arrange(q2$plot2)

	# Holding the amount of noise fixed, as the training dataset increases, the likelihood of the of superfluous features chosen to predict y-hat decreases. That is, we are able to more robustly
	# tease out the true relationship between y and x. Again, as before, holding the amount of noise fixed, as K increases, then algoritim uses more features to predict y-hat; thus the likelihood that
	# the features are spurious decreases, giving a lower MSE. Please note the above graphs.

########
# Question 2
########
	
	########
	# Part 1
	########

	# pairs(UC)
	summary(UC)

	########
	# Part 2
	########

	set.seed(1)
	ntrain <- round(NROW(UC) * 0.75) # sample 75% of training dataset
	wtrain <- sample(1:NROW(UC), ntrain) #these are the training rows
	
	train <- UC[wtrain]
	test <- UC[-wtrain]

	########
	# Part 3
	########

	base.plot <- ggplot(data = train, aes(x = mileage, y = price)) + 
						geom_point(color = "darkgrey", size = 1, alpha = 3 / 5) # show the relationship

	linear.plot <- 	base.plot + 
					geom_smooth(method = "lm", col = 'blue', linetype = "dashed", show.legend = TRUE) + # lm(price ~ mileage, data = train)
					geom_text(aes(x = 1e5, y = 1e4, label = 'Linear Fit', sep = ""), vjust = 4, size = 4, color = "blue") +
					ylim(0, 85000)
							
	print(linear.plot)

	########
	# Part 4
	########

	findBestPoly <- function(FUN, P = 1:15, data = train){ #gets best polynomial 
					
					poly <- list() # add a poly term with each succession
					for(p in P){ 
						cv.error <-  eval(FUN) # FUN must return MSE
						poly[[p]] <- cv.error
					}

					MSEcv <- cbind.data.frame(P = P, MSE = do.call(rbind, poly)) #log for better scaling
					optimal.poly <- MSEcv[which.min(MSEcv$MSE), ]$P # polynomial which minimizes cv.error
					return(list(MSEcv = MSEcv, optimal.P = optimal.poly))
	}

	poly.fit <- findBestPoly(FUN = quote(cv.glm(train, glm(price ~ poly(mileage, p, raw = TRUE), data = train), K = 5)$delta[1])) # find best polynomial line by cv
	glm.best <- glm(price ~ poly(mileage, poly.fit$optimal.P, raw = TRUE), data = train) # fit the best glm given optimal polynomial from cv
	glm.yhat <- predict(glm.best, test) # get glm predictions for test set

	polyMSEs <- as.vector(NULL) # get OOS MSE for all non-optimal polynomial degrees
	for(p in 1:15){ 
		glm.given <- glm(price ~ poly(mileage, p, raw = TRUE), data = train) # fit the glm given polynomial p
		glm.est <- predict(glm.given, test) # get glm predictions for test set
		polyMSEs[p] <- mean((glm.est - test[,price]) ^ 2)# OOS MSE 
	}

### TO DO something is not right with MSEs ?? why so large, normal?? 

	glm.mse.plot <- ggplot(data = as.data.frame(polyMSEs), aes(x = 1:15, y = polyMSEs)) + geom_point(color = "red", size = 4, alpha = 3 / 5) + xlab("log(MSE)") + ggtitle("GLM Polynomials vs OOS MSE")
	print(glm.mse.plot)

	poly.plot <- 	linear.plot + 
					geom_smooth(method = "lm", formula = y ~ poly(x, poly.fit$optimal.P, raw = TRUE), col = 'red', linetype = "solid") + # plot the optimal polynomial fit on all the data
					geom_text(aes(x = 4e5, y = 3e4, label = paste('Optimal Polynomial Term:', poly.fit$optimal.P, "\n", "(GLM)", sep = ""), sep = ""), vjust = 4, size = 4, color = "red") +
					ggtitle("Best fit Linear & Polynomial models on training data")
	print(poly.plot)

	########
	# Part 5
	########

	## KNN ## // caret package implementation of cv.knn is too slow; use custom

		knn.fit <- docvknn(x = as.data.frame(train$mileage), y = as.vector(train$price), k = 1:50, nfold = 5) #get MSE for each K given 5 folds	
		knn <- kknn(formula = price ~ mileage, train = train, test = test, kernel = "rectangular", k = which.min(knn.fit)) # fit knn to all training data given best k
		knn.yhat <- data.frame(fit = knn$fitted.values)

		knnMSEs <- as.vector(NULL) # get OOS MSE for all non-optimal Ks
		for(k in 1:50){ 
			knn.given <- kknn(formula = price ~ mileage, train = train, test = test, kernel = "rectangular", k = k) # fit the glm given polynomial p
			knn.est <- as.data.frame(knn.given$fitted.values) # get glm predictions for test set
			knnMSEs[k] <- mean((as.matrix(knn.est) - test[ ,price]) ^ 2) # OOS MSE 
		}

## TO DO MSEs are fucked up # the observations have to been divided 
		knn.mse.plot <- ggplot(data = as.data.frame(knnMSEs), aes(x = 1:50, y = knnMSEs)) + geom_point(color = "red", size = 4, alpha = 3 / 5) + xlab("Ks") + ggtitle("KNN(Ks) vs OOS MSE") + ylab("MSE")
		print(knn.mse.plot)

		knn.plot <- 	poly.plot + 
						geom_line(data = knn.yhat, aes(y = fit, x = test[ ,mileage]), col = "green") +
						geom_text(aes(x = 2e5, y = 4e4, label = paste('Optimal K:', which.min(knn.fit), "\n", "(KNN)", sep = ""), sep = ""), vjust = 4, size = 4, color = "green") +
						ggtitle("Best fit Linear & Polynomial & KNN & Tree models on testing data")
		print(knn.plot)

	## Tree ##

		tree.fit <- rpart(price ~ mileage, data = train, control = rpart.control(minsplit = 5,  cp = 0.0001, xval = 5)) #xval is 5 fold cross validation; and allow for complex tree
		best.cp <- tree.fit$cptable[which.min(tree.fit$cptable[ ,"xerror"]), "CP"] # complexity parameter that minimizes MSE
		best.tree <- prune(tree.fit, cp = best.cp)
		tree.yhat <- data.frame(fit = predict(best.tree, test))

		print(tree.fit$cptable[which.min(tree.fit$cptable[ ,"xerror"]), ]) # best tree parameters

		treeMSEs <- as.vector(NULL) # get OOS MSE for all non-optimal trees
		for(t in 1:NROW(tree.fit$cptable)){ 
			given.cp <- tree.fit$cptable[t, "CP"]
			given.tree <- prune(tree.fit, cp = given.cp)
			tree.est <- predict(given.tree, test)
			treeMSEs[t] <- mean((tree.est - test[,price]) ^ 2)# OOS MSE 
		}

		tree.mse.plot <- ggplot(data = as.data.frame(treeMSEs), aes(x = 1:NROW(tree.fit$cptable), y = treeMSEs)) + 
								geom_point(color = "red", size = 4, alpha = 3 / 5) + 
								xlab("complexity Parameter") + ggtitle("Tree(Cp) vs OOS MSE") + ylab("MSE")
		print(tree.mse.plot)

		tree.plot <- 	knn.plot + 
						geom_line(data = tree.yhat, aes(y = fit, x = test[ ,mileage]), col = "orange") + 
						geom_text(aes(x = 2e5, y = 6e4, label = paste('Optimal Complexity Parameter:', round(best.cp, 5), "\n", "(CART)", sep = ""), sep = ""), vjust = 4, size = 4, color = "orange") +
						ggtitle("Best fit Linear & Polynomial & KNN & Tree models on testing data")
		print(tree.plot)

		best.univariate <- which.min(data.frame(best.tree = min(treeMSEs), best.knn = min(knnMSEs), best.poly = min(polyMSEs))) # model that has lowest OOS MSE; choose this model 
		print(best.univariate) # this is the best univarite model

	########
	# Part 6
	########

	scale.xs <- function(x) {return((x - min(x)) / (max(x) - min(x)))} #scale the xs
	train[ , ':=' (	mileage = scale.xs(mileage), year = scale.xs(year))]
	test[ , ':=' (	mileage = scale.xs(mileage), year = scale.xs(year))]

	## KNN ##

		knn.fit <- docvknn(	x = train[ ,c('mileage', 'year'), with = FALSE], # multivariate version
							y = as.vector(train$price), k = 1:50, nfold = 5) #get MSE for each K given 5 folds

		knn <- kknn(formula = price ~ year + mileage, 	
					train = train[ ,c('price','mileage', 'year'), with = FALSE], 
					test = test[ ,c('price','mileage', 'year'), with = FALSE], 
					kernel = "rectangular", 
					k = which.min(knn.fit)) # fit knn to all training data given best k

		knn.yhat <- data.frame(fit = knn$fitted.values)

		knnMSEs <- as.vector(NULL) # get OOS MSE for all non-optimal Ks
		for(k in 1:50){ 
			knn.given <- kknn(formula = price ~ year + mileage, 
								train = train[ ,c('price','mileage', 'year'), with = FALSE], 
								test = test[ ,c('price','mileage', 'year'), with = FALSE], 
								kernel = "rectangular", k = k) # fit the glm given polynomial p

			knn.est <- as.data.frame(knn.given$fitted.values) # get glm predictions for test set
			knnMSEs[k] <- mean((as.matrix(knn.est) - test[ ,price]) ^ 2) # OOS MSE 
		}

## TO DO MSEs are fucked up
		knn.mse.plot <- ggplot(data = as.data.frame(knnMSEs), aes(x = 1:50, y = knnMSEs)) + geom_point(color = "red", size = 4, alpha = 3 / 5) + xlab("Ks") + ggtitle("KNN(Ks) vs OOS MSE") + ylab("MSE")
		print(knn.mse.plot)
		which.min(knn.fit) # optimal K

	## Tree ##

		tree.fit <- rpart(	price ~ mileage + year, 
							data = train[ ,c('price','mileage', 'year'), with = FALSE], 
							control = rpart.control(minsplit = 5,  cp = 0.0001, xval = 5)) #xval is 5 fold cross validation; and allow for complex tree
		best.cp <- tree.fit$cptable[which.min(tree.fit$cptable[ ,"xerror"]), "CP"] # complexity parameter that minimizes MSE
		best.tree <- prune(tree.fit, cp = best.cp)
		tree.yhat <- data.frame(fit = predict(best.tree, test))

		print(tree.fit$cptable[which.min(tree.fit$cptable[ ,"xerror"]), ]) # best tree parameters

		treeMSEs <- as.vector(NULL) # get OOS MSE for all non-optimal trees
		for(t in 1:NROW(tree.fit$cptable)){ 
			given.cp <- tree.fit$cptable[t, "CP"]
			given.tree <- prune(tree.fit, cp = given.cp)
			tree.est <- predict(given.tree, test)
			treeMSEs[t] <- mean((tree.est - test[ ,price]) ^ 2)# OOS MSE 
		}

		tree.mse.plot <- ggplot(data = as.data.frame(treeMSEs), aes(x = 1:NROW(tree.fit$cptable), y = treeMSEs)) + 
								geom_point(color = "red", size = 4, alpha = 3 / 5) + 
								xlab("complexity Parameter") + ggtitle("Tree(Cp) vs OOS MSE") + ylab("MSE")
		print(tree.mse.plot)

		best.multivariate <- which.min(data.frame(best.tree = min(treeMSEs), best.knn = min(knnMSEs))) # model that has lowest OOS MSE; choose this model
		print(best.multivariate) # this is the best multivariate model, compare to univariate

	########
	# Part 7
	########

	tree.fit <- rpart(price ~ ., data = train, control = rpart.control(minsplit = 5,  cp = 0.0001, xval = 5)) #xval is 5 fold cross validation; and allow for complex tree
	best.cp <- tree.fit$cptable[which.min(tree.fit$cptable[ ,"xerror"]), "CP"] # complexity parameter that minimizes MSE
	best.tree <- prune(tree.fit, cp = best.cp)
	tree.yhat <- data.frame(fit = predict(best.tree, test))
	MSE <- mean((tree.yhat - test[ ,price]) ^ 2)
	print(MSE) # MSE of tree fit to all features

	########
	# BONUS
	########

	# Explanation: In order to find the most relevant variables we look towards the toward the complexity parameter output found in the tree output. We note the tabulated results that indicate
	# how much each split contributes to improving the 'fit' of the tree model. These are the most important variables. We could now isolate these splits and their respective variables. Then, we
	# could use these variabls as the inputs to a more simple, interactive linear model. Because these variables are the most important explanatory features in the dataset, our interacted linear model 
	# should now predict better than a naive linear or polynomial model

	print(tree.fit$cptable) # aka isolate top 6 variables and use them in an interacted linear regression model