import numpy as np
import scipy
import os.path
f = open("Tower-Large.txt","r")
array = np.zeros((1024,1024))
i= 0
for line in f:
    if i == 0:
    	i =1
    else:
    	print i
        curLine = line.strip().split(" ")
        array[i-1,:] = np.array(curLine)
        i +=1
result= np.absolute(np.fft.fft2(array))
np.savetxt("Tower-Large-fft2d.txt",result,delimiter=" ",fmt="%4d")
