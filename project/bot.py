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

# Загрузка модели
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Отправьте любое обращение")

@bot.message_handler(func=lambda message: True)
def classify_message(message):
    text_vec = vectorizer.transform([message.text])
    prediction = model.predict(text_vec)[0]
    bot.reply_to(message, f"📝 Категория: {category_translation[prediction]}")

@app.route('/')
def health():
    return "OK", 200

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
    bot.process_new_updates([update])
    return "OK", 200

if __name__ == '__main__':
    # Устанавливаем webhook
    render_url = os.environ.get('RENDER_EXTERNAL_URL')
    if render_url:
        bot.set_webhook(url=f"{render_url}/{BOT_TOKEN}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
