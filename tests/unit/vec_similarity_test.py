import unittest
import vec_similarity as similarity

class SimilarityTest(unittest.TestCase):

  def test_manhatten_distance(self):
    point1 = (5, 10, 15)
    point2 = (5, 7, 11)
    dist = similarity.manhatten_distance(point1, point2)
    self.assertEqual(dist, 7)

  def test_euclidean_distance(self):
    point1 = (5, 10, 15)
    point2 = (5, 7, 11)
    dist = similarity.euclidean_distance(point1, point2)
    self.assertEqual(dist, 5)

  def test_pearson_coefficient(self):
    point1 = (1, 2, 3)
    point2 = (4, 5, 6)
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 1)

    point1 = (1, 2, 3)
    point2 = (0, 2, 4)
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 1)

    point1 = (1, 2, 3)
    point2 = (3, 2, 1)
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, -1)

    point1 = (1, 2, 3)
    point2 = (2, 2, 2)
    dist = similarity.pearson(point1, point2)
    self.assertEqual(dist, 0)

  def test_cosine_similarity(self):
    point1 = (1, 2, 3)
    point2 = (1, 2, 3)
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 1)

    point1 = (1, 2, 3)
    point2 = (2, 4, 6)
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 1)

    point1 = (0, 0, 1)
    point2 = (0, 1, 0)
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 0)

    point1 = (1, 0, 1)
    point2 = (0, 1, 1)
    dist = similarity.cosine(point1, point2)
    self.assertEqual(dist, 0.5)

  def test_nearest_manhatten_distance(self):
    users = { "John": [1, 2, 3],
              "Sarah": [1, 2, 4],
              "Andrew": [5, 2, 1] }
    nearest = similarity.nearest_manhatten("John", users)
    self.assertEqual(nearest, [(1.0, 'Sarah'), (6.0, 'Andrew')])

  def test_nearest_euclidean_distance(self):
    users = { "John": [1, 2, 3],
              "Sarah": [1, 2, 4],
              "Andrew": [5, 2, 0] }
    nearest = similarity.nearest_euclidean("John", users)
    self.assertEqual(nearest, [(1.0, 'Sarah'), (5.0, 'Andrew')])

  def test_nearest_pearson_coefficient(self):
    users = { "John": [1, 2, 3],
              "Sarah": [0, 2, 4],
              "Andrew": [3, 2, 1] }
    nearest = similarity.nearest_pearson("John", users)
    self.assertEqual(nearest, [(0.0, 'Sarah'), (2.0, 'Andrew')])

  def test_nearest_cosine_similarity(self):
    users = { "John": [1, 2, 3],
              "Sarah": [1, 2, 3],
              "Andrew": [-1, -2, -3] }
    nearest = similarity.nearest_cosine("John", users)
    self.assertEqual(nearest, [(0.0, 'Sarah'), (2.0, 'Andrew')])

if __name__ == '__main__':
  unittest.main()