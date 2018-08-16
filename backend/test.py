import json

import numpy as np



foo = range(5)

print(foo)

arr = np.array([0,0])

for i in range(10):
  foo = np.array([i,i])
  arr = np.vstack((arr, foo))

to_delete = [1,4,5]
to_choose = [x for x in range(10) if x not in to_delete]
print(arr[to_choose, :])
print(np.delete(arr, to_delete))


test =np.array(np.meshgrid([4, 5], [0, 1])).T.reshape(-1,2)
print(arr[test])

