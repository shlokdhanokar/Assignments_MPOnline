# Unsupervised Learning and Reinforcement Learning

## Clustering

Clustering groups similar records without any labels. K-Means partitions data into k clusters by alternately assigning each point to its nearest centroid and recomputing the centroids. It requires k to be chosen in advance and assumes clusters are roughly round and similarly sized.

The elbow method plots the within-cluster sum of squares against k and looks for the point where the curve bends. The silhouette score measures how similar a point is to its own cluster compared with the nearest other cluster, and can confirm or contradict the elbow.

## Dimensionality reduction

Principal Component Analysis finds the orthogonal directions of greatest variance and projects the data onto the first few. It removes correlated dimensions while retaining most of the variance, which makes high-dimensional data visualisable in two dimensions and often speeds up downstream models.

Feature scaling is essential before PCA and before any distance-based method such as K-Means, SVM or KNN, because a feature measured on a larger numeric scale would otherwise dominate the distance calculation purely because of its units.

## Recommendation systems

Collaborative filtering recommends items based on the behaviour of similar users or the similarity between items, learned from a user-item rating matrix. It can surface an item that shares no attributes with anything the user has seen, but it cannot score an item nobody has rated — the cold-start problem.

Content-based filtering recommends items whose attributes resemble those the user already liked. It handles new items naturally because it needs no ratings, but it cannot discover interests outside the user's existing profile.

Matrix factorisation decomposes the sparse rating matrix into low-rank user and item factors, filling in missing entries. A recommender is judged on the ranked list it produces, so ranking metrics such as Precision at k and NDCG matter more than rating-prediction error.

## Reinforcement learning

In reinforcement learning an agent interacts with an environment, takes actions, and receives rewards. Its goal is to learn a policy that maximises cumulative discounted reward. Unlike supervised learning, there are no labelled correct actions — the agent must discover them through trial and error.

Q-learning learns the expected return of taking an action in a state. Deep Q-Networks approximate that value function with a neural network, which requires two stabilising components: an experience replay buffer that decorrelates consecutive transitions, and a target network that holds the regression target still while the online network chases it.

Double DQN reduces the over-estimation bias of the plain maximum operator by letting the online network choose the next action and the target network score it.

The exploration-exploitation trade-off is managed with an epsilon-greedy policy, which takes a random action with probability epsilon and the best known action otherwise, decaying epsilon over training.

Deep Q-Networks are not monotonic: a run can reach a good policy and then collapse, so the best-performing weights should be snapshotted during training rather than evaluating whatever the final episode leaves behind.

The main limitation of reinforcement learning is sample efficiency: an agent may need hundreds of thousands of interactions to learn a task a human picks up in minutes, which is why it is applied where simulation is cheap.
