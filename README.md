## RAGMusic

Music recommendation system based on RAG.

Direct setup (you will need api keys):
1) Run ```setup.py```
2) Run ```tgbot.py```

Docker setup
1) docker pull lunteg/ragmusic_tgbot:latest (https://hub.docker.com/r/lunteg/ragmusic_tgbot)
2) docker run -d --name music-bot-container -v {DB Location}:/app/data -e TELEGRAM_API="{TG KEY}" -e SB_AUTH_DATA="{GigaChat Key}" ragmusic_tgbot

Bot nickname in telegram is @MusicListenRus_bot 


### Data

В качестве датасета взят [корпус музыки 1950-2019 годов](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019/data), состоящий из 24000 треков и 5000 исполнителей. Основные жанры: поп, кантри.

### Metrics

Были выбраны как стандартные для рекомендательных систем (а задача здесь нами рассматривается именно в такой постановке), так и необычные: novelty и personalization, особенностью которых является то, что их можно посчитать без "ground truth", т.е. без разметки.