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
        self.conn = sqlite3.connect("data.db")
        self.cursor = self.conn.cursor()
        self.df = pd.read_sql_query("SELECT * FROM my_table", self.conn)
        
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.embeddings = self.create_embeddings()
        
        # Настройка FAISS
        self.index = self.create_faiss_index(self.embeddings)

        # Настройка GigaChat
        giga_key = os.environ.get("SB_AUTH_DATA")
        self.giga = GigaChat(
            credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)

        # Настройка Telegram-бота
        bot_token = os.environ.get("TELEGRAM_API")
        app = ApplicationBuilder().token(bot_token).build()

        # Команда /recommend
        app.add_handler(CommandHandler("recommend", self.recommend))
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
    def get_user_history(self, user_id, returning_field = 'track_id'):
        if returning_field == 'track_id':
            self.cursor.execute(
                'SELECT track_id FROM history WHERE user_id = ?', (user_id,))
            rows = self.cursor.fetchall()
        elif returning_field == 'track_name':
            self.cursor.execute(
                'SELECT track_name FROM history WHERE user_id = ?', (user_id,))
            rows = self.cursor.fetchall()

        return [row[0] for row in rows]
    
    def create_embeddings(self):
        """Создает TF-IDF векторизацию описаний треков."""
        descriptions = self.df["lyrics"].fillna("")  # Используем текст описаний или лирики
        embeddings = self.vectorizer.fit_transform(descriptions)
        return embeddings.toarray()
    
    def create_faiss_index(self, embeddings):
        """Создает FAISS индекс."""
        # Создаем FAISS индекс для векторов L2
        index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # Добавляем embeddings в индекс
        embeddings = embeddings.astype(np.float32)  # FAISS требует тип данных float32
        index.add(embeddings)
        return index

    # Функция для поиска похожих треков
    def search_similar_tracks(self, query, top_k=5):
        """Ищет похожие треки через FAISS."""
        query_vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        distances, indices = self.index.search(query_vector, k=top_k)
        
        # Извлекаем индексы найденных треков
        similar_indices = indices[0]
        return self.df.iloc[similar_indices]

    # Основной обработчик команды для поиска песни
    async def find_track(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_input = update.message.text
        similar_tracks = self.search_similar_tracks(user_input, top_k=5)

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
        
        
    async def recommend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id

        # Получаем историю в виде индексов
        listening_history = self.get_user_history(user_id)

        if not listening_history:
            await update.message.reply_text("У вас нет истории поиска.")
            return

        print(listening_history)
        # Генерация контекста для рекомендаций
        context = self.df.iloc[listening_history]
        print(context.iterrows())
        recommendations = self.generate_recommendations(context.iterrows())
        await update.message.reply_text(recommendations)
    
    def generate_recommendations(self, context):
        """Генерирует текстовые рекомендации через GigaChat."""
        structured_context = "\n\n".join(
        [
            f"Песня {i + 1}:\n"
            f"Название: {row['track_name']}\n"
            f"Исполнитель: {row['artist_name']}\n"
            f"Жанр: {row.get('genre', 'не указан')}\n"
            f"Текст: {row['lyrics'][:500]}..."  # Ограничиваем текст лирики
            for i, row in context
        ]
    )
        system_context = (
            f"Вы музыкальный бот, который предлагает рекомендации на основе песен. "
            f"Вот список песен:\n\n{structured_context}\n\n"
            f"Рекомендуйте похожие песни, основываясь на жанре, тексте и настроении."
        )
        response = self.giga.invoke(system_context)
        return response.content


if __name__ == '__main__':
    main = MusicBot()
