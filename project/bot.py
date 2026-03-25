import telebot
import pickle
import os

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

# Инициализируем бота с вашим токеном
BOT_TOKEN = '8692035566:AAGi2EyGRD8y67pLF4mNy1AJXNC6g6X1M_E'
bot = telebot.TeleBot(BOT_TOKEN)

# Обрабатываем команду /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Отправьте любое обращение, чтобы я мог классифицировать его")

# Обрабатываем обращения
@bot.message_handler(func=lambda message: True)
def classify_message(message):
    user_text = message.text
    text_vec = vectorizer.transform([user_text]) # Преобразуем текст так же, как при обучении
    prediction = model.predict(text_vec)[0] # Получаем предсказание от модели
    bot.reply_to(message, f"📝 Категория сообщения: {category_translation[prediction]}") # Отправляем результат пользователю

# Запускаем бота
print("Бот запущен")
bot.infinity_polling()