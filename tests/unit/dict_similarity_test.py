import unittest
import dict_similarity as similarity

class SimilarityTest(unittest.TestCase):

  def test_manhatten_distance(self):
    point1 = {"a" : 5, "b" : 10, "c" : 15}
    point2 = {"a" : 5, "b" : 7, "c" : 11}
    dist = similarity.manhatten_distance(point1, point2)
    self.assertEqual(dist, 7)

  def test_euclidean_distance(self):
    point1 = {"a" : 5, "b" : 10, "c" : 15}
    point2 = {"a" : 5, "b" : 7, "c" : 11}
    dist = similarity.euclidean_distance(point1, point2)
    self.assertEqual(dist, 5)

    point1 = {"a" : 5, "b" : 10, "c" : 15}
    point2 = {"a2" : 5, "b2" : 7, "c2" : 11}
    dist = similarity.euclidean_distance(point1, point2)
    self.assertEqual(dist, 0)

  def test_pearson_coefficient(self):
    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a" : 4, "b" : 5, "c" : 6}
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 1)

    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a" : 0, "b" : 2, "c" : 4}
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 1)

    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a" : 3, "b" : 2, "c" : 1}
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, -1)

    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a" : 2, "b" : 2, "c" : 2}
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 0)

    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a2" : 2, "b2" : 2, "c2" : 2}
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 0)

  def test_cosine_similarity(self):
    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a" : 1, "b" : 2, "c" : 3}
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 1)

    point1 = {"a" : 1, "b" : 2, "c" : 3}
    point2 = {"a" : 2, "b" : 4, "c" : 6}
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 1)

    point1 = {"a" : 0, "b" : 0, "c" : 1}
    point2 = {"a" : 0, "b" : 1, "c" : 0}
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 0)

    point1 = {"a" : 1, "b" : 0, "c" : 1}
    point2 = {"a" : 0, "b" : 1, "c" : 1}
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 0.5)

  def test_nearest_manhatten_distance(self):
    users = { "John": {"a" : 1, "b" : 2, "c" : 3},
              "Sarah": {"a" : 1, "b" : 2, "c" : 4},
              "Andrew": {"a" : 5, "b" : 2, "c" : 1} }
    nearest = similarity.nearest_manhatten("John", users)
    self.assertEqual(nearest, [(1.0, 'Sarah'), (6.0, 'Andrew')])

  def test_nearest_euclidean_distance(self):
    users = { "John": {"a" : 1, "b" : 2, "c" : 3},
              "Sarah": {"a" : 1, "b" : 2, "c" : 4},
              "Andrew": {"a" : 5, "b" : 2, "c" : 0} }
    nearest = similarity.nearest_euclidean("John", users)
    self.assertEqual(nearest, [(1.0, 'Sarah'), (5.0, 'Andrew')])

  def test_nearest_pearson_coefficient(self):
    users = { "John": {"a" : 1, "b" : 2, "c" : 3},
              "Sarah": {"a" : 0, "b" : 2, "c" : 4},
              "Andrew": {"a" : 3, "b" : 2, "c" : 1} }
    nearest = similarity.nearest_pearson("John", users)
    self.assertEqual(nearest, [(0.0, 'Sarah'), (2.0, 'Andrew')])

  def test_nearest_cosine_similarity(self):
    users = { "John": {"a" : 1, "b" : 2, "c" : 3},
              "Sarah": {"a" : 1, "b" : 2, "c" : 3},
              "Andrew": {"a" : -1, "b" : -2, "c" : -3} }
    nearest = similarity.nearest_cosine("John", users)
    self.assertEqual(nearest, [(0.0, 'Sarah'), (2.0, 'Andrew')])

if __name__ == '__main__':
  unittest.main()