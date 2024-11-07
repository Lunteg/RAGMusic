import sqlite3
import numpy as np
import pandas as pd

conn = sqlite3.connect("data.db")
sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.int32, lambda val: int(val))
cursor = conn.cursor()

# Создание таблицы истории, если она не существует
cursor.execute('''
    DROP TABLE IF EXISTS history;
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        user_id INTEGER,
        track_id INTEGER,
        track_name TEXT
    )
''')
conn.commit()

df = pd.read_csv("data/music_dataset.csv")
df.to_sql("my_table", conn, if_exists='replace', index=False)
