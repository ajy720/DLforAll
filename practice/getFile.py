import numpy as np
from os.path import join, dirname, abspath

iris_dict = {
  0 : 'Iris-setosa',
  1 : 'Iris-versicolor',
  2 : 'Iris-virginica'
}

iris_dict_reverse = {
  'Iris-setosa' : 0.,
  'Iris-versicolor' : 1.,
  'Iris-virginica' : 2.
}

def getIris(file):
  
  f = open(join(dirname(abspath(__file__)), file))
  iris = f.readlines()
  f.close()

  x_data = []
  y_data = []

  for data in iris[1:]:
    data = data.strip('\n').split(',')[1:]
    x_data.append([*map(np.float32, data[:-1])])
    y_data.append([np.float32(iris_dict_reverse[data[-1]])])

  x_data = np.array(x_data)
  y_data = np.array(y_data)

  return x_data, y_data