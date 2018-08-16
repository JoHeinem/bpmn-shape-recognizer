import json

import numpy as np



foo = range(5)

print(foo)

arr = np.array([0,0])

for i in range(10):
  foo = np.array([i,i])
  arr = np.vstack((arr, foo))

to_delete = [1,4,5]
foo = list(range(0,10))
to_choose = [x for x in range(len(foo)) if x not in to_delete]

print(foo[to_choose])


#
# print(arr[to_choose, :])
# print(np.delete(arr, to_delete))
#
#
# test =np.array(np.meshgrid([4, 5], [0, 1])).T.reshape(-1,2)
# print(arr[test])
#
