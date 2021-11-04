# Recommendation_Sys
A simple recommendation system based on Yelp-dataset

In this project, I have created a hybrid recommendation system based on Yelp dataset. I have used the weighted average of my item-based recommendations and model-based recommendations. Using CF my system is focussing on the neighbors of the item while the model based system is focussing on both neighbors and items, to solve the cold start.

For item based CF- For users and businesses that were missing (cold-start), I gave them rating of 3.0 to keep an average rating. Then I created a list of co-rated users and businesses, and used Pearson's correlation to find similarity between neighbors and get recommendations.

For Model Based system- I selected additional features from extra data files provided and used XGBoost library and implemented linear regression to output recommendations.

Finally, I combined the two systems using weighted average to produce a more efficient recommendation system. This method helps to give prediction score for unrated items that are impossible to be recommended by only using collaborative filtering.
