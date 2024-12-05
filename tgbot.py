import pandas as pd
import numpy as np
from langchain.vectorstores import FAISS
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.chat_models.gigachat import GigaChat
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.ext import CallbackQueryHandler
import os
import sqlite3


class MusicBot:
    user_history = {}

    def __init__(self):
        self.conn = sqlite3.connect("/app/data/data.db")
        self.cursor = self.conn.cursor()
        self.df = pd.read_sql_query("SELECT * FROM my_table", self.conn)

        # Индексы и векторизаторы
        self.title_artist_vectorizer = TfidfVectorizer(max_features=5000)
        self.feature_vectorizer = None  # Векторизация характеристик

        # Создание эмбеддингов
        self.title_artist_embeddings = self.create_title_artist_embeddings()
        self.feature_embeddings = self.create_feature_embeddings()

        # Создание FAISS индексов
        self.title_artist_index = self.create_faiss_index(
            self.title_artist_embeddings)
        self.feature_index = self.create_faiss_index(self.feature_embeddings)

        # Настройка GigaChat
        giga_key = os.environ.get("SB_AUTH_DATA")
        self.giga = GigaChat(
            credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)

        # Настройка Telegram-бота
        bot_token = os.environ.get("TELEGRAM_API")
        app = ApplicationBuilder().token(bot_token).build()

        # Команда /start
        app.add_handler(CommandHandler("start", self.start))
        # Команда /help
        app.add_handler(CommandHandler("help", self.help_command))
        # Команда /recommend
        app.add_handler(CommandHandler("recommend", self.recommend))
        # Команда /history
        app.add_handler(CommandHandler("history", self.history))
        
        # Прием названия песни от пользователя
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self.find_track))
        app.add_handler(CallbackQueryHandler(self.track_selection_handler))

        app.run_polling(close_loop=False)

    # Функция для добавления трека в историю пользователя
    def add_to_history(self, user_id, track_name, track_id):

        self.cursor.execute(
            'INSERT INTO history (user_id, track_name, track_id) VALUES (?, ?, ?)', (user_id, track_name, track_id))
        self.conn.commit()

    # Функция для получения истории пользователя
    def get_user_history(self, user_id, returning_field='track_id'):
        if returning_field == 'track_id':
            self.cursor.execute(
                'SELECT track_id FROM history WHERE user_id = ?', (user_id,))
            rows = self.cursor.fetchall()
        elif returning_field == 'track_name':
            self.cursor.execute(
                'SELECT track_name FROM history WHERE user_id = ?', (user_id,))
            rows = self.cursor.fetchall()

        return [row[0] for row in rows]
    
    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "Доступные команды:\\n"
            "/start - Начать работу с ботом\\n"
            "/help - Показать это меню помощи\\n"
            "/recommend - Показать рекомендации\\n"
            "Просто отправьте название трека, чтобы начать собирать историю для рекомендаций!"
        )
        await update.message.reply_text(help_text)
    
    def create_title_artist_embeddings(self):
        """Создаёт TF-IDF векторизацию по названию и автору."""
        data = (self.df["track_name"] + " " +
                self.df["artist_name"]).fillna("")
        embeddings = self.title_artist_vectorizer.fit_transform(data)
        return embeddings.toarray()

    def create_feature_embeddings(self):
        """Создаёт эмбеддинги на основе музыкальных характеристик."""
        feature_columns = [
            "lyrics", "genre", "dating", "violence", "shake the audience", "family/gospel",
            "romantic", "communication", "obscene", "music", "movement/places",
            "light/visual perceptions", "family/spiritual", "like/girls", "sadness",
            "feelings", "danceability", "loudness", "acousticness", "instrumentalness",
            "valence", "energy", "topic", "age"
        ]

        # Заполняем пропуски и объединяем характеристики
        features = self.df[feature_columns].fillna(
            "").astype(str).apply(" ".join, axis=1)

        self.feature_vectorizer = TfidfVectorizer(max_features=10000)
        embeddings = self.feature_vectorizer.fit_transform(features)
        return embeddings.toarray()

    def create_faiss_index(self, embeddings):
        """Создает FAISS индекс."""
        # Создаем FAISS индекс для векторов L2
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Добавляем embeddings в индекс
        # FAISS требует тип данных float32
        embeddings = embeddings.astype(np.float32)
        index.add(embeddings)
        return index

    # Функция для поиска похожих треков
    def search_by_title_and_artist(self, query, top_k=5):
        """Ищет треки по названию и автору."""
        query_vector = self.title_artist_vectorizer.transform(
            [query]).toarray().astype(np.float32)
        distances, indices = self.title_artist_index.search(
            query_vector, k=top_k)
        return self.df.iloc[indices[0]]
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /start.
        """
        welcome_message = (
            "👋 Добро пожаловать в MusicBot! 🎶\n\n"
            "Вот что я умею:\n"
            "1️⃣ Отправьте название песни, чтобы начать собирать историю для ваших рекомендаций.\n"
            "2️⃣ Используйте команду /recommend для получения рекомендаций на основе вашей истории.\n"
            "3️⃣ Используйте команду /history, чтобы посмотреть историю прослушиваний.\n\n"
            "Попробуйте прямо сейчас отправить название песни, чтобы начать! 🚀"
        )
        await update.message.reply_text(welcome_message)
        
    # Основной обработчик команды для поиска песни
    async def find_track(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_input = update.message.text
        similar_tracks = self.search_by_title_and_artist(user_input, top_k=5)

        if similar_tracks.empty:
            await update.message.reply_text("Не удалось найти похожие песни.")
            return

        # Создаем кнопки для выбора песни
        buttons = [
            [
                InlineKeyboardButton(
                    f"{row['track_name']} - {row['artist_name']}",
                    callback_data=str(index),
                )
            ]
            for index, row in similar_tracks.iterrows()
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(
            "Выберите песню из списка:", reply_markup=reply_markup
        )

    async def track_selection_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        track_id = int(query.data)
        track = self.df.iloc[track_id]

        # Сохраняем выбор в истории
        user_id = query.from_user.id
        self.add_to_history(user_id, track["track_name"], track_id)

        # Отправляем информацию о треке
        response = f"Вы выбрали песню:\nНазвание: {track['track_name']}\nИсполнитель: {track['artist_name']}"
        await query.edit_message_text(text=response)

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id
        listening_history = self.get_user_history(user_id, 'track_name')

        response = "История прослушивания:\n"
        await update.message.reply_text(response + '\n'.join(listening_history))

    def recommend_based_on_features(self, history_track_ids, top_k=10):
        """Рекомендует треки на основе музыкальных характеристик."""
        if not history_track_ids:
            return pd.DataFrame()

        # Получаем эмбеддинги для треков из истории
        history_embeddings = self.feature_embeddings[history_track_ids]
        user_profile_vector = np.mean(
            history_embeddings, axis=0).reshape(1, -1).astype(np.float32)

        # Ищем треки, похожие на профиль пользователя
        distances, indices = self.feature_index.search(user_profile_vector, k=top_k + len(history_track_ids))
        
        # Исключаем треки из истории
        recommended_indices = [idx for idx in indices[0] if idx not in history_track_ids]
        
        # Урезаем до `top_k` после фильтрации
        recommended_indices = recommended_indices[:top_k]
        
        return self.df.iloc[recommended_indices]
        
    async def recommend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id

        # Получаем историю в виде индексов
        listening_history = self.get_user_history(user_id)

        if not listening_history:
            await update.message.reply_text("У вас нет истории поиска.")
            return

        # Ищем похожие треки на основе истории
        similar_tracks = self.recommend_based_on_features(
            listening_history, top_k=5)

        if similar_tracks.empty:
            await update.message.reply_text("Не удалось найти похожие треки.")
            return

        recommendations = self.generate_recommendations(
            similar_tracks.iterrows())
        await update.message.reply_text(recommendations)

    def generate_recommendations(self, similar_tracks):
        """Генерирует текстовые рекомендации через GigaChat."""
        # Генерация контекста для рекомендаций
        structured_context = "\n\n".join(
            [
                f"Название: {row['track_name']}\n"
                f"Исполнитель: {row['artist_name']}\n"
                f"Жанр: {row.get('genre', 'не указан')}\n"
                f"Текст: {row['lyrics'][:500]}..."  # Ограничиваем текст лирики
                for _, row in similar_tracks
            ]
        )
        # Формируем системный запрос
        system_context = (
            f"Вы музыкальный бот, который предлагает рекомендации на основе песен."
            f"Вот список песен похожих на то, что слушает пользователь:\n\n{structured_context}\n\n"
            f"Ты должен предложить ему песни из этого списка и объяснить почему."
            f"Ответ должен быть без вопросов, вступления и заключения, содержать ТОЛЬКО список треков с твоими объяснениями."
        )
        print(structured_context)

        response = self.giga.invoke(system_context)
        return response.content


if __name__ == '__main__':
    main = MusicBot()
