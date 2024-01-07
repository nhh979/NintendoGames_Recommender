# NintendoGames_Recommender (In progess)

- In this project, we scraped over 2000 Nintendo Switch games on [Wikipedia](https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(0%E2%80%939_and_A)) using Beautiful Soup. We selected only that games that contain a hyperlink to their wiki page. We could then access to the link and scrap the information about the Gameplay and Plot for each game.
![](https://github.com/nhh979/NintendoGames_Recommender/blob/master/Images/WikiExample.png)
  
- Next, we cleaned the dataset by applying some filling missing value techniques, dropping unnecessary rows and columns, and cleaning text data with regular expressions.
  
- In the modeling phase, we first transformed the text data in the Gameplay column into the TF-IDF matrix using module `nltk` and `TfidfVectorizer` method from `Sklearn` module. From the TF-IDF matrix, we calculated the similarity distances between the texts by subtracting the `cosine_similarity`, which is from submodule `sklearn.metrics.pairwise`, from 1.  
  
- Finally, we could suggest the games that are similar to a given game by querying the games that have smallest similarity distance to the given game. For example, below are 5 games that are similar to **"The Legend of Zelda: Tears of the Kingdom"** with their respective similarity distance:  
  
![](https://github.com/nhh979/NintendoGames_Recommender/blob/master/Images/RecommendedGames.png)
