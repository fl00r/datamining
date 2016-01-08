def minkowski(dict1, dict2, r):
  dist = 0

  for k, v1 in dict1.items():
    v2 = dict2.get(k)
    if v2 is not None:
      dist += abs(v1 - v2) ** r

  return dist ** (1./r)

def manhatten_distance(dict1, dict2):
  return minkowski(dict1, dict2, 1)

def euclidean_distance(dict1, dict2):
  return minkowski(dict1, dict2, 2)


def mean(d):
  return sum(d.values()) / len(d)

def pearson(dict1, dict2):
  mean1 = mean(dict1);
  mean2 = mean(dict2);

  num = 0
  denum1 = 0
  denum2 = 0

  for k, x in dict1.items():
    y = dict2.get(k)
    if y is not None:
      x_ = (x - mean1)
      y_ = (y - mean2)
      num += x_ * y_
      denum1 += x_ ** 2
      denum2 += y_ ** 2

  denum = (denum1 * denum2) ** .5

  if denum == 0:
    return 0
  else:
    return num / denum

def zero_based_pearson(dict1, dict2):
  return -pearson(dict1, dict2) + 1

def vector_length(d):
  return sum(x ** 2 for x in d.values()) ** .5

def vector_product(dict1, dict2):
  p = 0
  for k, x in dict1.items():
    y = dict2.get(k)
    if y is not None:
      p += x * y

  return p

def cosine(dict1, dict2):
  length1 = sum(x ** 2 for x in dict1.values())
  length2 = sum(x ** 2 for x in dict2.values())

  if length1 == 0 or length2 == 0:
    raise Exception("You can't calculate cosine for vector with length 0")

  product = vector_product(dict1, dict2)

  return product / (length1 * length2) ** .5

def zero_based_cosine(dict1, dict2):
  return -cosine(dict1, dict2) + 1

def nearest(_key, _map, distance_f):
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