##############
# Homework 3 #
##############

########
# Load Config Files
########

	options("width" = 250)
	options(scipen  = 999)
	options(digits  = 003)

	library(ggplot2); require(gridExtra); library(MASS); library(Matrix); library(parallel)
	library(kknn); library(boot); library(rpart); library(data.table); library(h2o); library(caret)
	library(gamlr); library(BayesTree); library(xgboost); library(ranger); library(mxnet); library(tensorflow)

	set.seed(666) # the devils seed

	dir <- '/home/nmatare/projects/machine_learning/homework3'
	setwd(dir)

########
# Question 2
########

	download.file(paste("https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/", "HumanActivityRecognitionUsingSmartphones/ParseData.R", sep = ""), "ParseData.R")
	source("ParseData.R")
	data <- parse_human_activity_recog_data()

	trainX <- data.matrix(data$X_train)
	trainY <- as.integer(factor(data$y_train)) - 1

	testX <- data.matrix(data$X_test)
	testY <- as.integer(factor(data$y_test)) - 1
	testYnames <- levels(data$y_test)

	mx.set.seed(666)

	# Configure FNN architecture
	net = mx.symbol.Variable("data")

	net = mx.symbol.FullyConnected(net, name = "layer1", num_hidden = 1024) # hidden layer 
	net = mx.symbol.Activation(net, name = "actv1", act_type = "relu") # activation function [relu, tanh, sigmoid, softrelu]
	net = mx.symbol.Dropout(net, p = 0.10, name = "drop1") 

	net = mx.symbol.FullyConnected(net, name = "layer2", num_hidden = 1024) # hidden layer
	net = mx.symbol.Activation(net, name = "actv2", act_type = "relu") # activation function
	net = mx.symbol.Dropout(net, p = 0.10, name = "drop2")

	net = mx.symbol.FullyConnected(net, name = "layer3", num_hidden = 512) # hidden layer
	net = mx.symbol.Activation(net, name = "actv3", act_type = "relu") # activation function
	net = mx.symbol.Dropout(net, p = 0.10, name = "drop3")

	net = mx.symbol.FullyConnected(net, name = "layerOut", num_hidden = 6) # output layer; if classification, must be equal to # of classes
	architecture = mx.symbol.SoftmaxOutput(net, name = "sm") # softmax for classification

  	# model <- mx.model.FeedForward.create(
   # 				X = trainX, 
   # 				y = trainY, 
 		# 		eval.metric = mx.metric.accuracy,
   # 				verbose = TRUE,
   # 				symbol = architecture, # network architecture
                
   # 				ctx = mx.gpu(1),
   #              # ctx = mx.gpu(3), # mx.cpu or mx.gpu
   #              # kvstore = "local_allreduce_device", # http://mxnet-tqchen.readthedocs.io/en/latest/system/multi_node.html

			# 	optimizer = "adadelta", # ['sgd','adadelta'] # Stochastic Gradient Descent optimizer 
   #            	# optimizer = "sgd",
   #              # learning.rate = 0.05, # SGD step size [0 - 1]; how large of steps until local optimimum
   #              # momentum = 0.3, # SGD momentum (prevents getting stuck in local optimium)
                
   #              initializer = mx.init.uniform(0.01),
   #              array.layout = "rowmajor",
   #              array.batch.size = 1000, # the number of training examples in one forward/backward pass #large batch size, greater amount of data used in one iteration
   #              # begin.round = 1,
   #              num.round = 20, # number of epochs
   #              # epoch.end.callback = mx.callback.save.checkpoint(prefix = "learn.tmp", period = 1)
   #              # epoch.end.callback = mx.callback.early.stop(eval.metric = 0.30) # end early if train or eval metric goes below threshold
    #  )	

  	trainNeuralNet.GPU <- function(GPUs, epochs, X, Y, architecture, loss.function, optimizer, batch.size, seed = 666, verbose = TRUE){
  					# GPU memory efficent mxnet wrapper; saves 1 iteration to disk and cycles through GPUs
  					require(mxnet)
  					system('export MXNET_BACKWARD_DO_MIRROR=1') # default = 0; slower but doesn't make copies on GPU
					system('export MXNET_EXEC_NUM_TEMP=0') 

  					gpu.device <- 0 # init gpu counter
  					for(i in 1:epochs){

  						mx.set.seed(seed)
  						if(i == 1){ # init model at first epoch

  							model <- mx.model.FeedForward.create(
  										X = X, y = Y, symbol = architecture,
  										eval.metric = loss.function,
  										verbose = FALSE, ctx = mx.gpu(gpu.device),
  										optimizer = optimizer, begin.round = 1,
  										num.round = 1, array.batch.size = batch.size,
  										array.layout = 'rowmajor',
  										epoch.end.callback = mx.callback.save.checkpoint(prefix = "learn.tmp", period = 1)
  									)

  						} else if(i != 1){

  							old.model <- mx.model.load("learn.tmp", 1) # load last periods model
  							model <- mx.model.FeedForward.create(
  										X = X, y = Y, symbol = old.model$symbol,
  										eval.metric = loss.function,
  										verbose = FALSE, ctx = mx.gpu(gpu.device),
  										optimizer = optimizer, 
  										num.round = 1, array.batch.size = batch.size,
  										array.layout = 'rowmajor',
  			               				arg.params = old.model$arg.params, # load old layer weights and biases
  										aux.params = old.model$aux.params,  # load old layer extra params
  										epoch.end.callback = mx.callback.save.checkpoint(prefix = "learn.tmp", period = 1)
  									)

  						}

	  					gpu.device <- gpu.device + 1; gc()
	  					if(gpu.device > (GPUs - 1)) gpu.device <- 0 # cycle back to first cpu
		
  						if(verbose) print(paste("=== Epoch", i, "of", epochs, "==="))
  					}
  							
	  				remove <- capture.output({sapply(list.files(pattern = "*.params"), file.remove)}) # delete temp files created 	
  					return(model)
  	}

  	model <- trainNeuralNet.GPU(
  		GPUs = 4, 
  		epochs = 100, 
  		batch.size = 128,
  		X = trainX, Y = trainY, 
  		architecture = architecture,
  		optimizer = 'adadelta',
  		loss.function = mx.metric.accuracy
  	)

    # graph.viz(model$symbol, type = 'vis', direction = 'LR') # output of network architecture
	
	phats <- round(t(predict(model, testX, array.layout = "rowmajor")), 4)
	colnames(phats) <- testYnames
	head(phats); tail(phats)
	preds <- apply(phats, 1, which.max) - 1

	conf.mat <- try(confusionMatrix(testY, preds))
	1 - conf.mat$overall['Accuracy'] # error # best 4.58%

  	# Configure RNN architecture
	model <- mx.lstm(
				train.data = list(data = t(trainX), label = t(binY)),
				ctx = mx.cpu(),
				update.period = 1,
				num.lstm.layer = 2, 
				num.hidden = 16, 
				
				num.embed = 477, 
				seq.len = 477,
				num.label = 477,
				input.size = 477,

				num.round = 99, 
				batch.size = 32, 
				initializer = mx.init.uniform(0.1), 
				learning.rate = 0.05,
				clip_gradient = 1
	)

	## DeepWater h2o w/ mxnet ## need install from BoothIT
	h2o.init(nthreads = detectCores() -1)

	if (!h2o.deepwater_available()) return()

	trainX <- as.h2o(data$X_train, destination_frame = "Xtrain")
	trainY <- as.h2o(data$y_train, destination_frame = "Ytrain")

	# final model
	final_model <- h2o.deepwater(
					y = colnames(trainY), 
					training_frame = h2o.cbind(trainY, trainX), 
					input_dropout_ratio = 0.2,
					backend = 'mxnet',
					network = 'lenet',
					epochs = 20
	) 

	# Base h2o 
	h2oServer <- h2o.init(nthreads = detectCores() -1) # init server

	trainX <- as.h2o(data$X_train, destination_frame = "Xtrain")
	trainY <- as.h2o(data$y_train, destination_frame = "Ytrain")

	testX <- as.h2o(data$X_test, destination_frame = "Xtest")
	testY <- as.h2o(data$y_test, destination_frame = "Ytest")

	# try multiple models
	findBestParams <- function(neurons, layers, epochs = 10, activation = c("Tanh", "RectifierWithDropout", "Maxout", "MaxoutWithDropout"), dropout = c(0.2), l1.regularization = c(0, 1e-5)){

					hidden <- rep(neurons, layers) # number of neurons per layer creats the hidden layers
					params <- expand.grid(epochs, activation, dropout, l1.regularization)
					names(params) <- c("epochs", "activation", "input_dropout_ratio", "l1")
					params <- apply(t(params), 2, as.list)

					i <- 1
					while(TRUE){ # add hidden layers to each parameter
						params[[i]]$epochs <- as.numeric(params[[i]]$epochs)
						params[[i]]$input_dropout_ratio <- as.numeric(params[[i]]$input_dropout_ratio)
						params[[i]]$l1 <- as.numeric(params[[i]]$l1)
						params[[i]]$hidden <- hidden
						i <- i + 1
						if(i > length(params)) break
					}

					return(params)
	}

	gridSearchDNN <- function(param, x, y, training_frame, testing_frame){
					# param: parameter values to try
					# training_frame: h2o data.frame that contains both x and y
					# testing_frame: h2o data.frame that contains both oos x and y

					arguments <- modifyList(list(x = x, y = y, training_frame = training_frame, nfolds = 0, seed = 666, reproducible = TRUE), param) # CV set to zero, no CV 

					model <- do.call(h2o.deeplearning, arguments); 

					perform <- h2o.performance(model, testing_frame)
					confusion.mat <- h2o.confusionMatrix(perform)$Error
					total.error <- confusion.mat[length(confusion.mat)] # last one is the total error
					return(total.error)
	}

	gridSearchDNN.wrapper <- function(params, x, y, training_frame, testing_frame){

					results <- list()
					for(n in 1:length(params)){

						print(paste("Starting parameter", n, "of", length(params)))
						error <- tryCatch({gridSearchDNN(params[[n]], x = x, y = y, training_frame = training_frame, testing_frame = testing_frame)}, error = function(e) {return(NA)}) 
						results[[n]] <- error

					}
					return(do.call(rbind, results))
	}

	params1 <- findBestParams(neurons = 100, layers = 4)
	out1 <- gridSearchDNN.wrapper(params = params1, x = 1:NCOL(trainX), y = NCOL(trainX) + 1, training_frame = h2o.cbind(trainX, trainY), testing_frame = h2o.cbind(testX, testY))

	params2 <- findBestParams(neurons = 500, layers = 4)
	out2 <- gridSearchDNN.wrapper(params = params2, x = 1:NCOL(trainX), y = NCOL(trainX) + 1, training_frame = h2o.cbind(trainX, trainY), testing_frame = h2o.cbind(testX, testY))


	params3 <- findBestParams(neurons = 500, layers = 8)
	out3 <- gridSearchDNN.wrapper(params = params3, x = 1:NCOL(trainX), y = NCOL(trainX) + 1, training_frame = h2o.cbind(trainX, trainY), testing_frame = h2o.cbind(testX, testY))
	
	params4 <- findBestParams(neurons = 1000, layers = 8)
	out4 <- gridSearchDNN.wrapper(params = params4, x = 1:NCOL(trainX), y = NCOL(trainX) + 1, training_frame = h2o.cbind(trainX, trainY), testing_frame = h2o.cbind(testX, testY))

	# final model
	final_model <- h2o.deeplearning(y = colnames(trainY), 
									training_frame = h2o.cbind(trainY, trainX), 
									activation = "RectifierWithDropout",
									input_dropout_ratio = 0.2,
									l1 = 1e-5,
									hidden = c(500, 500, 500, 500),
									mini_batch_size = 1,
									epochs = 20,
									# nfolds = 5, 
				  					# stopping_tolerance = 0.005, # stop if not 0.5% in moving average of 5 rounds 
  									stopping_rounds = 0
					) 

	# Check out what is validation_frame?
	perform <- h2o.performance(final_model, h2o.cbind(testX, testY))
	confusion.mat <- h2o.confusionMatrix(perform)$Error
	total.error <- confusion.mat[length(confusion.mat)] # last one is the total error
	total.error
	
	# beat 0.0468
	test = h2o.predict(final_model, testX)
