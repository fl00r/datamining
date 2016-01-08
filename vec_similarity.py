def check_cardinality(vec1, vec2):
  if len(vec1) != len(vec2):
    raise Exception("Cardinality of two vectors doesn't match")

def minkowski(vec1, vec2, r):
  """Generalization of Manhatten and Euclidean distances.
  The greater the `r`, the more a large difference in one dimension 
  will influence the total difference.
  """

  check_cardinality(vec1, vec2)

  dist = sum(abs(x - y) ** r for x, y in zip(vec1, vec2))

  return dist ** (1./r)

def manhatten_distance(vec1, vec2):
  """
  >>> manhatten_distance((0, 1, 2), (0, 1, 2))
  0
  >>> manhatten_distance((0, 5, 10), (0, 4, 8))
  3
  >>> manhatten_distance((0, 5), (0, 4, 8))
  Exception: wrong cardinality
  """

  return minkowski(vec1, vec2, 1)

def euclidean_distance(vec1, vec2):
  """
  >>> euclidean_distance((0, 1, 2), (0, 1, 2))
  0
  >>> euclidean_distance((0, 5, 10), (0, 5, 8))
  2
  >>> euclidean_distance((0, 5), (0, 4, 8))
  Exception: wrong cardinality
  """

  return minkowski(vec1, vec2, 2)


def mean(vec):
  """
  >>> mean([1,2,3,4])
  2.5
  """
  return sum(vec) / len(vec)

def pearson(vec1, vec2):
  """Pearson correlation coefficient
  >>> pearson([1, 2, 3], [2, 4, 6])
  1
  >>> pearson([1, 2, 3], [3, 2, 1])
  -1
  """
  check_cardinality(vec1, vec2)

  vec1_mean = mean(vec1);
  vec2_mean = mean(vec2);

  num = 0
  denum1 = 0
  denum2 = 0

  for x, y in zip(vec1, vec2):
    x_ = (x - vec1_mean)
    y_ = (y - vec2_mean)
    num += x_ * y_
    denum1 += x_ ** 2
    denum2 += y_ ** 2

  denum = (denum1 * denum2) ** .5

  if denum == 0:
    return 0
  else:
    return num / denum

def zero_based_pearson(vec1, vec2):
  """For distance unification (less is better)
  >>> pearson([1,2], [1,2])
  1
  >>> zero_based_pearson([1,2], [1,2])
  0
  >>> pearson([1,2], [2,1])
  -1
  >>> zero_based_pearson([1,2], [2,1])
  2
  """
  return -pearson(vec1, vec2) + 1

def vector_length(vec):
  return sum(x ** 2 for x in vec) ** .5

def vector_product(vec1, vec2):
  return sum(x * y for x, y in zip(vec1, vec2))

def cosine(vec1, vec2):
  """ Cosine similarity
  >>> cosine([1,0], [0,1])
  0
  >>> cosine([1,2], [2,4])
  1
  >>> cosine([1,2], [0, 0])
  Exception: zero length
  """
  check_cardinality(vec1, vec2)

  vec1_length = sum(x ** 2 for x in vec1)
  vec2_length = sum(x ** 2 for x in vec2)

  if vec1_length == 0 or vec2_length == 0:
    raise Exception("You can't calculate cosine for vector with length 0")

  product = vector_product(vec1, vec2)

  return product / (vec1_length * vec2_length) ** .5

def zero_based_cosine(vec1, vec2):
  """For distance measure unification (less is better)
  """
  return -cosine(vec1, vec2) + 1

def nearest(_key, _map, distance_f):
  """_map is a dict of `key` => `vec`
  _key is a target vector in _map
  distance_f is a distance function
  returns _map keys sorted by their distance to _key
  """
  vec = _map[_key]
  return sorted((distance_f(vec, v), k) for k,v in _map.items() if k != _key)

def nearest_manhatten(_key, _map):
  return nearest(_key, _map, manhatten_distance)

def nearest_euclidean(_key, _map):
  return nearest(_key, _map, euclidean_distance)

def nearest_pearson(_key, _map):
  return nearest(_key, _map, zero_based_pearson)

def nearest_cosine(_key, _map):
  return nearest(_key, _map, zero_based_cosine)