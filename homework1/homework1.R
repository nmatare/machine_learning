
##############
# Homework 1 #
##############

########
# Load Config Files
########

	library(ggplot2)
	require(gridExtra)
	library(MASS)
	library(kknn)

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
					linear <- lm(y ~ x, data = simulation$train) #fit linear model
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

	question2 <- doQuestionOne(quote(3 + exp(x + 1)))

	question2$plot1; dev.new()
	grid.arrange(question2$plot2); dev.new()
	question2$plot3

	########
	# Simulation 3
	########

	question3 <- doQuestionOne(quote(2 + sin(2 * x)), noise = 0)
	
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

	# Bonus Question:
	q1 <- doQuestionOne(num.train = 100, noise = 5, equation = quote(2 + sin(2 * x))) # given training dataset with 100
	grid.arrange(q1$plot2); dev.new()

	q2 <- doQuestionOne(num.train = 1000, noise = 5, equation = quote(2 + sin(2 * x))) # given training dataset with 100
	grid.arrange(q2$plot2)

	# Holding the amount of noise fixed, as the training dataset increases, the likelihood of the of superfluous features chosen to predict y-hat decreases. That is, we are able to more robustly
	# tease out the true relationship between y and x. Again, as before, holding the amount of noise fixed, as K increases, then algoritim uses more features to predict y-hat; thus the likelihood that
	# the features are spurious decreases, giving a lower MSE. Please note the above graphs.
	
########
# Question 2
########
