import sqlite3
import math
from collections import defaultdict

from metrics import novelty, personalization, ILS

# Соединение с SQLite базой данных
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Выполнение SQL-запроса для получения данных
cursor.execute("""
SELECT user_id, GROUP_CONCAT(track_name) AS recommended_tracks
FROM history
GROUP BY user_id;
""")

# Подготовка списка рекомендаций
recommendations = []
user_ids = []
for row in cursor.fetchall():
    user_id = row[0]
    user_ids.append(user_id)
    recommended_tracks = row[1].split(',')  # Преобразуем строку в список
    recommendations.append(recommended_tracks)

# Закрытие соединения с базой данных
conn.close()

novelty_scores = novelty(recommendations)
# ILS_scores = ILS(recommendations)
personalization_scores = personalization(recommendations)

# Вывод результатов
print('user_id: \t', user_ids)
print('novelty: \t', novelty_scores)
# print('ILS: ', novelty_scores)
print('personalization:', novelty_scores)