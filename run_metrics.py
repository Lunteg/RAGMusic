import sqlite3

from metrics import novelty, personalization, mean_ap_at_k, mean_ndcg_at_k

conn = sqlite3.connect('data/data.db')
cursor = conn.cursor()

cursor.execute("""
SELECT user_id, GROUP_CONCAT(track_name) AS recommended_tracks
FROM history
GROUP BY user_id;
""")

user_ids = []
recommendations = []
for row in cursor.fetchall():
    user_id = row[0]
    user_ids.append(user_id)
    recommended_tracks = row[1].split(',')
    recommendations.append(recommended_tracks)

conn.close()

novelty_scores = novelty(recommendations)
personalization_scores = personalization(recommendations)

# print('user_id: \t', user_ids)
print('novelty: \t', novelty_scores)
print('personalization:', personalization_scores)
print(mean_ap_at_k([["a", "b", "c"]], [["a", "b"]], 10))
print(mean_ndcg_at_k([["a", "b", "c"]], [["a", "b"]], 10))
