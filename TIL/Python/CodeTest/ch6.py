arr = [['송진호',12],['김재연',94],['이정은',92]]
print(sorted(arr,key=lambda a: a[1]))

#['송진호',12]
#['김재연',94]
#['이정은',92]


'''
import random
import time
start = time.time()
list=[]
for _ in range(100000):
	list.append(random.random())


print("time :", time.time() - start)
'''