
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

	makeSimulation <- function(equation, noise = 0){
					# noise controls the number of superfurlous predictor vars added to dataset

					# create training dataset
					x <- rnorm(100, mean = 0, sd = 1) #predictor x
					x.noise <- NULL					
					if(noise > 0) x.noise <- as.data.frame(sapply(rep(100, noise), function(w) rnorm(w, mean = 0, sd = 1))) # generate noise variables from normal distribution

					e <- rnorm(100, mean = 0, sd = 1) # random noise
					y <- eval(equation) + e # y is the function plus some noise

					train <- as.data.frame(cbind(y, x, x.noise)) #train models on predictor and noise
		
					test <- list() # create testing dataset
					for(i in 1:100){ #do 100 times to get 10,000 observations

						x <- rnorm(100, mean = 0, sd = 1)
						e <- rnorm(100, mean = 0, sd = 1) # random noise
						y <- eval(equation) + e
						test[[i]] <- cbind.data.frame(y = y, x = x) # true model has no noise so don't include it in the test set
					}

					test <- as.data.frame(do.call(rbind, test))
					stopifnot(isTRUE(dim(test)[1] == 10000)) # sanity check
					return(list(train = train, test = test))
	}

	makeKnn <- function(data, K){ #wrapper function for Knn
					knn <- kknn(		formula = y ~ ., # where . is everything or x or x1, x2, x3, etc
										train = data$train, # train on relationship between x and y
										test = data$test, # given x, find y 
										kernel = "rectangular", 
										k = K)
					return(knn$fitted.values)
	}

	doQuestionOne <- function(equation, p1 = TRUE, p2 = TRUE, p3 = TRUE, ...){

					#p1/p3 control verbosity of print
					simulation <- makeSimulation(equation)

					# Part 2 and 3
					base.plot <- ggplot(	data = simulation$test, aes(x = x, y = y)) + 
											geom_point(color = "darkgrey", size = 1, alpha = 3 / 5) + # show the relationship
											stat_function(fun = function(x) eval(equation), colour = "black") + # show the true linear equation
											geom_text(aes(x = max(x) - 1, y = 8, label = 'True Function', sep = ""), vjust = -4, size = 4, color = "black") # add text
					plot(base.plot)


					plot1 <- 	base.plot + ggtitle("Linear Model")	+		
								geom_smooth(method = "lm", col = 'blue', linetype = "dashed", show.legend = TRUE) + # show the true line; this is essentially lm(y ~ x, data = simulation)
								geom_text(aes(x = max(x) - 1, y = 8, label = 'Linear Fit', sep = ""), vjust = 4, size = 4, color = "blue")

					if(p1) print(plot1); dev.new()

					# Part 4
					knn2fitted <- 	cbind.data.frame(fit = makeKnn(K = 2, data = simulation), x = simulation$test$x) #fit on training data and then output the fitted y values and corresponding x's
					knn12fitted <- 	cbind.data.frame(fit = makeKnn(K = 12, data = simulation), x = simulation$test$x)

					plot.knn2 <- base.plot + geom_line(data = knn2fitted, aes(y = fit, x = x), col = "green") + ggtitle("KNN at K = 2")
					plot.knn12 <- base.plot + geom_line(data = knn12fitted, aes(y = fit, x = x), col = "blue") + ggtitle("KNN at K = 12")

					if(p2) plot2 <- grid.arrange(plot.knn2, plot.knn12, ncol = 2); dev.new()

					# Part 5
					MSEs <- list()
					for(k in 2:15){
						fitted <- makeKnn(K = k, data = simulation) # get fitted vals
						MSEs[[k]] <- mean((simulation$test$x - fitted) ^ 2) # get MSE
					}

					MSEs <- cbind.data.frame(K = -log(1 / rep(2:15, 1)), MSE = do.call(rbind, MSEs)) # get MSEs for each K (K here is really -log(1/K))
					linear <- lm(y ~ x, data = simulation$train) #fit linear model
					pred <- predict(linear, simulation$test)
					MSE.lin <- mean((simulation$test$x - pred) ^ 2)

					plot3 <- ggplot(		data = MSEs, aes(x = K, y = MSE)) + geom_point() + # plot MSE of all Ks
											geom_hline(yintercept = MSE.lin, col = "blue", linetype = "dashed") + # linear MSE
											xlab("-log(1/K)") +
											ggtitle("MSEs of KNNs") 
									
					if(p3) print(plot3)
	}

	########
	# Simulation 1
	########

	doQuestionOne(equation = quote(2 + 1.8 * x), noise = 0)

	########
	# Simulation 2
	########

	doQuestionOne(quote(3 + exp(x + 1)))

	########
	# Simulation 3
	########

	doQuestionOne(quote(2 + sin(2 * x)))

	########
	# Simulation 4
	########

	grid.arrange(nrow = 4, ncol = 5)
	for(n in 1:20){

		doQuestionOne(quote(2 + sin(2 * x)), noise = 20, p1 = FALSE, p2 = FALSE) # only print MSE charts

		draw_plot(plot, x = 0, y = 0, width = 1, height = 1)
	}


