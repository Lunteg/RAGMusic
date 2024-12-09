## RAGMusic

Music recommendation system based on RAG.

Steps:
1) Run ```setup.py```
2) Run ```tgbot.py```

For Docker
1) docker pull lunteg/ragmusic_tgbot:latest (https://hub.docker.com/r/lunteg/ragmusic_tgbot)
2) docker run -d --name music-bot-container -v {DB Location}:/app/data -e TELEGRAM_API="{TG KEY}" -e SB_AUTH_DATA="{GigaChat Key}" ragmusic_tgbot

Bot nickname in telegram is @MusicListenRus_bot 
