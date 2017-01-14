
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

	doQuestionOne <- function(equation){

					makeSimulation <- function(equation){

									# create training dataset
									x <- rnorm(100, mean = 0, sd = 1)
									y <- eval(equation) + rnorm(100, mean = 0, sd = 1) #y is the function plust some noise
									train <- cbind.data.frame(y, x)
						
									test <- list() # create testing dataset
									for(i in 1:100){ #do 100 times to get 10,000 observations

										test[[i]] <- cbind(	y = eval(equation) + rnorm(100, mean = 0, sd = 1), # why is the function plus noise
															x =  rnorm(100, mean = 0, sd = 1) # x is from norm(0,1)
														)
									}

									test <- as.data.frame(do.call(rbind, test))
									stopifnot(isTRUE(dim(test)[1] == 10000)) # sanity check
									return(list(train = train, test = test))
					}

					simulation <- makeSimulation(equation)

					# Part 2 and 3
					base.plot <- ggplot(	data = simulation$test, aes(x = x, y = y)) + 
											geom_point(color = "darkgrey", size = 1, alpha = 3 / 5) + # show the relationship
											geom_abline(intercept = 2, slope = 1.8, color = "black") + # show the true linear equation
											geom_text(aes(x = max(x) - 1, y = 8, label = 'True Function', sep = ""), vjust = -4, size = 4, color = "black") # add text

					plot1 <- 	base.plot + ggtitle("Linear Model")	+		
								geom_smooth(method = "lm", col = 'blue', linetype = "dashed", show.legend = TRUE) + # show the true line; this is essentially lm(y ~ x, data = simulation)
								geom_text(aes(x = max(x) - 1, y = 8, label = 'Linear Fit', sep = ""), vjust = 4, size = 4, color = "blue")

					print(plot1); dev.new()

					# Part 4
					makeKnn <- function(data, K){ #wrapper function for Knn
									knn <- kknn(		formula = y ~ x, 
														train = data$train, # train on relationship between x and y
														test = data$test, # given x, find y 
														kernel = "rectangular", 
														k = K)
									return(knn$fitted.values)
					}

					knn2fitted <- 	cbind.data.frame(fit = makeKnn(K = 2, data = simulation), x = simulation$test$x) #fit on training data and then output the fitted y values and corresponding x's
					knn12fitted <- 	cbind.data.frame(fit = makeKnn(K = 12, data = simulation), x = simulation$test$x)

					plot.knn2 <- base.plot + geom_line(data = knn2fitted, aes(y = fit, x = x), col = "green") + ggtitle("KNN at K = 2")
					plot.knn12 <- base.plot + geom_line(data = knn12fitted, aes(y = fit, x = x), col = "blue") + ggtitle("KNN at K = 12")

					plot2 <- grid.arrange(plot.knn2, plot.knn12, ncol = 2); dev.new()

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
									
					print(plot3)
	}

	########
	# Simulation 1
	########

	doQuestionOne(quote(2 + 1.8 * x))

	########
	# Simulation 2
	########

	doQuestionOne(quote(3 + exp(x + 1) + e))

	########
	# Simulation 3
	########

	doQuestionOne(quote(2 + sin(2 * x) + e))

	########
	# Simulation 4
	########

	## Special case where X grows, will have to write custom function

	doQuestionOne()

	z * rnorm(1, mean = 0, sd = 0)

	2 + sin(2 * x) +  + e

	quote(sin(2 * x) + 2 + 0 * rnorm(1, mean = 0, sd = 0) + e)

	sin(2 * x) + 2 + 0 * rnorm(1, mean = 0, sd = 0) + e
	yi = sin(2xi1) + 2 + 0 × xi2 + · · · + 0 × xip + i