import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import pairwise_distances
import numpy as np

import sqlite3

from langchain.prompts import PromptTemplate
from langchain.chat_models.gigachat import GigaChat


from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from thefuzz import process

import os

class MusicBot:
    user_history = {}
    def __init__(self):
        self.conn = sqlite3.connect("data.db")
        self.cursor = self.conn.cursor()
        self.df = pd.read_sql_query("SELECT * FROM my_table", self.conn)
        
        giga_key = os.environ.get("SB_AUTH_DATA")
        self.giga = GigaChat(credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)
        
        bot_token = os.environ.get("TELEGRAM_API")  
        app = ApplicationBuilder().token(bot_token).build()

        app.add_handler(CommandHandler("recommend", self.recommend))  # Команда /recommend
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.find_track))  # Прием названия песни от пользователя
            
        app.run_polling(close_loop=False)
    
    # Функция для добавления трека в историю пользователя
    def add_to_history(self, user_id, track_name, track_id):

        self.cursor.execute('INSERT INTO history (user_id, track_name, track_id) VALUES (?, ?, ?)', (user_id, track_name, track_id))
        self.conn.commit()
        
    # Функция для получения истории пользователя
    def get_user_history(self, user_id):
        self.cursor.execute('SELECT track_id FROM history WHERE user_id = ?', (user_id,))
        rows = self.cursor.fetchall()

        return [row[0] for row in rows]
    
    # Функция для поиска похожих треков
    def find_similar_tracks(self, data, user_indices, n_neighbors=5):
        """
        Ищет похожие треки на основе треков, прослушанных пользователем.

        Parameters:
        - data (pd.DataFrame): Датасет с признаками треков.
        - user_tracks (list): Список индексов треков, прослушанных пользователем.
        - n_neighbors (int): Число похожих треков для вывода.

        Returns:
        - similar_tracks (pd.DataFrame): Датафрейм с похожими треками.
        """
        # Отбираем только числовые признаки
        feature_cols = ['len', 'dating', 'violence', 'world/life', 'sadness', 'feelings',
                        'danceability', 'loudness', 'acousticness', 'instrumentalness',
                        'valence', 'energy', 'age']
        X = data[feature_cols].values

        # Применяем случайные проекции для снижения размерности
        projector = SparseRandomProjection(n_components=10, random_state=42)
        X_projected = projector.fit_transform(X)

        # Если нет треков от пользователя, возвращаем пустой DataFrame
        if len(user_indices) == 0:
            return pd.DataFrame(columns=['artist_name', 'track_name'])

        # Находим средний вектор для треков пользователя
        user_vectors = X_projected[user_indices]
        avg_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

        # Вычисляем расстояния между средним вектором и всеми треками
        distances = pairwise_distances(avg_vector, X_projected, metric='euclidean')

        # Сортируем и выбираем n_neighbors ближайших треков
        similar_indices = np.argsort(distances[0])[:n_neighbors + 1]  # +1 чтобы исключить сам трек
        similar_indices = similar_indices[1:]  # Исключаем сам трек из результатов

        # Возвращаем DataFrame с похожими треками
        return data.iloc[similar_indices][['artist_name', 'track_name']]

    def generate_recommendations(self, user_tracks_id):
        user_tracks_list = self.df.iloc[user_tracks_id]
        
        # Найти похожие треки
        recommendations = self.find_similar_tracks(self.df, user_tracks_id)
            
        # Сгенерировать текстовые рекомендации с использованием RAG
        search_results = [f"{row['track_name']} by {row['artist_name']}" for _, row in recommendations.iterrows()]
        context = "Это рекомендуемая песни: " + "; ".join(search_results)
        
        system_context = """Вы - бот, который дает рекомендации по песням. 
        Вы отвечаете очень короткими предложениями и не добавляете лишней информации. Ответ должен быть на руском языке
        {context}
        """
    
        tracks_names = [f"{row['track_name']} by {row['artist_name']}" for _, row in user_tracks_list.iterrows()]
        query = f"Основываясь на песнях {tracks_names}, порекомендуйте похожие песни с описаниями."
        
        response = self.giga.invoke(system_context + query)
        return response.content

    def find_similar_track_id_by_name(self, title):
        df = self.df
        
        track_titles = df['track_name'].tolist()
        # Ищем ближайшее совпадение в списке названий
        closest_match = process.extractOne(title, track_titles)
        print('closest_match')
        print(closest_match)
        # Извлекаем индекс найденного названия в DataFrame
        closest_index = df[df['track_name'] == closest_match[0]].index[0]
        return closest_index, closest_match[0]


    # Основной обработчик команды для поиска песни
    async def find_track(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id
        user_input = update.message.text
        result_id, result_name = self.find_similar_track_id_by_name(user_input)
        if result_id is not None:
            result_id = int(result_id)
            track = self.df.iloc[result_id]

            response = f"Найдена похожая песня:\nНазвание: {track['track_name']}\nИсполнитель: {track['artist_name']}"
            await update.message.reply_text(response)
            
            # Добавляем найденный трек в историю пользователя
            self.add_to_history(user_id, track['track_name'], result_id)
        else:
            await update.message.reply_text("Песня не найдена, попробуйте другое название.")

    async def recommend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id
        listening_history = self.get_user_history(user_id)
        if listening_history:
            recommendations = self.generate_recommendations(listening_history)
            
            response = "Рекомендации на основе вашей истории:\n"
            response += recommendations
        else:
            response = "У вас пока нет истории прослушивания. Найдите трек, чтобы начать историю."

        await update.message.reply_text(response)

if __name__ == '__main__':
    main = MusicBot()
    