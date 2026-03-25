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

# Получаем токен из переменной окружения
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    print("❌ BOT_TOKEN не найден!")
    exit()

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Отправьте любое обращение, чтобы я мог классифицировать его")

@bot.message_handler(func=lambda message: True)
def classify_message(message):
    try:
        user_text = message.text
        text_vec = vectorizer.transform([user_text])
        prediction = model.predict(text_vec)[0]
        bot.reply_to(message, f"📝 Категория сообщения: {category_translation[prediction]}")
    except Exception as e:
        bot.reply_to(message, "❌ Ошибка обработки")
        print(f"Ошибка: {e}")

@app.route('/')
def health():
    return "OK", 200

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.method == 'POST':
        json_str = request.get_data().decode('UTF-8')
        update = telebot.types.Update.de_json(json_str)
        bot.process_new_updates([update])
        return "OK", 200
    return "Method not allowed", 405

if __name__ == '__main__':
    # Получаем URL сервиса
    # Render автоматически подставляет RENDER_EXTERNAL_URL
    render_url = os.environ.get('RENDER_EXTERNAL_URL')
    
    # Если переменная не установлена, используем localhost для теста
    if not render_url:
        print("⚠️ RENDER_EXTERNAL_URL не найден, пропускаем установку webhook")
    else:
        webhook_url = f"{render_url}/{BOT_TOKEN}"
        try:
            # Удаляем старый webhook и устанавливаем новый
            bot.remove_webhook()
            bot.set_webhook(url=webhook_url)
            print(f"✅ Webhook установлен: {webhook_url}")
        except Exception as e:
            print(f"❌ Ошибка установки webhook: {e}")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Бот запущен на порту {port}")
    app.run(host='0.0.0.0', port=port)
