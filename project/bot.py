import os
import pickle
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

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Модель успешно загружена")
except FileNotFoundError:
    print("Ошибка: файлы модели не найдены.")
    exit()

BOT_TOKEN = os.environ.get('BOT_TOKEN', '8692035566:AAGi2EyGRD8y67pLF4mNy1AJXNC6g6X1M_E')
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Отправьте любое обращение, чтобы я мог классифицировать его")

@bot.message_handler(func=lambda message: True)
def classify_message(message):
    user_text = message.text
    text_vec = vectorizer.transform([user_text])
    prediction = model.predict(text_vec)[0]
    bot.reply_to(message, f"📝 Категория сообщения: {category_translation[prediction]}")

@app.route('/')
def health():
    return "OK", 200

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
    bot.process_new_updates([update])
    return "OK", 200

if __name__ == '__main__':
    bot.remove_webhook()
    webhook_url = f"https://{os.environ.get('RENDER_EXTERNAL_URL', 'localhost')}/{BOT_TOKEN}"
    bot.set_webhook(url=webhook_url)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
