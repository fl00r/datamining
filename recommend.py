import dict_similarity as similarity

class Recommender1:
  def __init__(self, data, titles):
    self.data = data

  def recommend(self, k):
    return similarity.nearest_cosine(k, self.data)[0]
    

  @classmethod
  def from_file(self, fn):
    titles = []
    data = {}
    with open(fn) as f:
      for line in f:
        if len(titles) == 0:
          for title in line.split(",")[1:-1]:
            titles.append(title.strip("\""))
        else:
          k = None
          for i, v in enumerate(line.rstrip().split(",")):
            if i == 0:
              k = v.strip("\"")
              data[k] = {}
            else:
              if v is not None and v is not "":
                data[k][i] = int(v)

    return self(data, titles)

# recommender = Recommender1.from_file("./data/Movie_Ratings.csv")
# print(recommender.recommend("The Matrix"))
# print(recommender.recommend("Lord of the Rings"))
