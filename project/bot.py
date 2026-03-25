import os
import pickle
import sys
from flask import Flask, request
import telebot

app = Flask(__name__)

category_translation = {
    'account': 'Аккаунт',
    'other': 'Прочее',
    'billing': 'Оплата',
    'delivery': 'Доставка',
    'tech': 'Техподдержка'
}

# Загрузка модели
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    sys.exit(1)

# Получаем токен — обязательно через os.environ
BOT_TOKEN = os.environ.get('BOT_TOKEN')

if not BOT_TOKEN:
    print("❌ BOT_TOKEN не найден в переменных окружения!")
    print("Проверьте, что переменная BOT_TOKEN добавлена в Environment Variables")
    sys.exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Отправьте любое обращение, чтобы я мог классифицировать его")

@bot.message_handler(func=lambda message: True)
def classify_message(message):
    try:
        text_vec = vectorizer.transform([message.text])
        prediction = model.predict(text_vec)[0]
        bot.reply_to(message, f"📝 Категория: {category_translation[prediction]}")
    except Exception as e:
        bot.reply_to(message, "❌ Ошибка обработки")
        print(f"Ошибка: {e}")

@app.route('/')
def health():
    return "OK", 200

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    json_str = request.get_data().decode('utf-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200

if __name__ == '__main__':
    render_url = os.environ.get('RENDER_EXTERNAL_URL')
    if render_url:
        webhook_url = f"{render_url}/{BOT_TOKEN}"
        try:
            bot.remove_webhook()
            bot.set_webhook(url=webhook_url)
            print(f"✅ Webhook установлен: {webhook_url}")
        except Exception as e:
            print(f"⚠️ Ошибка установки webhook: {e}")
    else:
        print("⚠️ RENDER_EXTERNAL_URL не найден, webhook не установлен")

    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Бот запущен на порту {port}")
    app.run(host='0.0.0.0', port=port)
