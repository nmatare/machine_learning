test

def fizzBuzz(startingNumber, endingNumber, fizzMultiple, buzzMultiple):
	for i in range(startingNumber,endingNumber):
		a=''
		if i%(fizzMultiple)==0:
			a+='fizz'
		if i%(buzzMultiple)==0:
			a+='buzz'
		if a=='':
			a=i
		print a