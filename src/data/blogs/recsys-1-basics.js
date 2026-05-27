export const recsys_1_basics = {
  slug: 'recsys-1-basics',
  title: 'Recommender Systems: Netflix ও Amazon কীভাবে সাজেস্ট করে?',
  description: 'Collaborative Filtering, Content-Based Filtering, User-Item Matrix, Cosine Similarity এবং Pearson Correlation — রেকমেন্ডার সিস্টেমের মূল ভিত্তি।',
  date: 'মে ২০২৬',
  category: 'রেকমেন্ডার সিস্টেম',
  readTime: 13,
  content: `
    <h3>১. Recommender System কী?</h3>
    <p>একটি <strong>Recommender System</strong> user-এর past behavior, preferences এবং অন্য users-এর data ব্যবহার করে relevant items suggest করে।</p>
    <p>বাস্তব উদাহরণ:</p>
    <ul>
      <li><strong>Netflix:</strong> "আপনি যেহেতু Peaky Blinders দেখেছেন, Narcos দেখুন"</li>
      <li><strong>Amazon:</strong> "এই product কিনেছেন যারা তারা এটিও কিনেছেন"</li>
      <li><strong>Spotify:</strong> Discover Weekly playlist</li>
      <li><strong>YouTube:</strong> Up Next recommendations</li>
    </ul>
    <p>তিনটি প্রধান approach:</p>
    <ol>
      <li><strong>Collaborative Filtering:</strong> Similar users/items খুঁজে recommendation করা</li>
      <li><strong>Content-Based Filtering:</strong> Item features analysis করে similar items suggest করা</li>
      <li><strong>Hybrid:</strong> উভয় method combine করা</li>
    </ol>

    <h3>২. User-Item Matrix</h3>
    <p>Collaborative Filtering-এর ভিত্তি হলো <strong>User-Item Matrix</strong> — rows = users, columns = items, values = ratings।</p>
    <pre><code class="language-python">import numpy as np
import pandas as pd

# Sample user-movie ratings (0 = not rated)
ratings_data = {
    'User':    ['Alice', 'Alice', 'Bob',   'Bob',   'Carol', 'Carol', 'Dave',  'Dave'],
    'Movie':   ['Titanic','Inception','Titanic','Interstellar','Inception','Interstellar','Titanic','Inception'],
    'Rating':  [5, 4, 4, 5, 5, 4, 2, 3]
}
df = pd.DataFrame(ratings_data)

# Pivot to user-item matrix
matrix = df.pivot_table(index='User', columns='Movie', values='Rating', fill_value=0)
print(matrix)
print(f"\\nMatrix shape: {matrix.shape}")  # 4 users × 3 movies
print(f"Sparsity: {(matrix==0).sum().sum() / matrix.size:.1%}")  # How sparse
</code></pre>

    <h3>৩. User-Based Collaborative Filtering</h3>
    <p>যে users একই items rate করেছে তাদের similarity বের করে, similar users-এর ratings থেকে prediction করা হয়।</p>
    <pre><code class="language-python">from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, numpy as np

# Full ratings matrix
ratings = pd.DataFrame({
    'Alice':  [5, 4, 0, 0, 1],
    'Bob':    [4, 0, 4, 1, 0],
    'Carol':  [0, 0, 5, 4, 0],
    'Dave':   [0, 1, 0, 5, 4],
    'Eve':    [1, 0, 0, 4, 5],
}, index=['Titanic','Inception','Interstellar','The Matrix','Avengers'])

user_sim = pd.DataFrame(
    cosine_similarity(ratings.T),
    index=ratings.columns, columns=ratings.columns
)
print("User Similarity Matrix:")
print(user_sim.round(3))

def predict_rating(user, movie, ratings, user_sim, n_neighbors=2):
    if ratings.loc[movie, user] != 0:
        return ratings.loc[movie, user]  # Already rated
    # Find users who rated this movie
    rated_users = ratings.columns[ratings.loc[movie] != 0].tolist()
    if not rated_users: return 3.0  # default
    # Get similarity scores
    sims = user_sim.loc[user, rated_users]
    top_users = sims.nlargest(n_neighbors)
    # Weighted average
    weighted_sum = sum(top_users[u] * ratings.loc[movie, u] for u in top_users.index)
    sim_sum = top_users.abs().sum()
    return round(weighted_sum / sim_sum if sim_sum > 0 else 3.0, 2)

# Predict Alice's rating for Interstellar (she hasn't seen it)
pred = predict_rating('Alice', 'Interstellar', ratings, user_sim)
print(f"\\nPredicted rating for Alice on Interstellar: {pred}")
</code></pre>

    <h3>৪. Item-Based Collaborative Filtering</h3>
    <p>Item-Based CF তুলনামূলকভাবে stable — items কম পরিবর্তন হয়, users-এর চেয়ে। Amazon এই approach ব্যবহার করে।</p>
    <pre><code class="language-python">item_sim = pd.DataFrame(
    cosine_similarity(ratings),
    index=ratings.index, columns=ratings.index
)

def recommend_items(user, ratings, item_sim, n=3):
    unrated = ratings.index[ratings[user] == 0].tolist()
    predictions = {}
    for item in unrated:
        rated_items = ratings.index[ratings[user] != 0].tolist()
        sims = item_sim.loc[item, rated_items]
        weights = sims * ratings.loc[rated_items, user]
        predictions[item] = weights.sum() / sims.abs().sum() if sims.abs().sum() > 0 else 0
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

recs = recommend_items('Alice', ratings, item_sim)
print("Recommendations for Alice:")
for movie, score in recs:
    print(f"  {movie}: {score:.2f}")
</code></pre>

    <h3>৫. Content-Based Filtering</h3>
    <pre><code class="language-python">from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

movies = pd.DataFrame({
    'title': ['Titanic', 'Inception', 'Interstellar', 'The Matrix', 'Avengers'],
    'genres': [
        'romance drama historical',
        'thriller sci-fi mind-bending',
        'sci-fi space drama time-travel',
        'sci-fi action philosophy',
        'superhero action sci-fi'
    ]
})

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

def content_recommend(title, n=3):
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [(movies['title'][i], round(s, 3)) for i, s in scores]

print("Content-based recs for Interstellar:")
for title, sim in content_recommend('Interstellar'):
    print(f"  {title}: {sim}")
</code></pre>
  `
};
