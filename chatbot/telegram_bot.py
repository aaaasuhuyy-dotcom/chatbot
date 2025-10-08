# Python 3.9+
import os
import pickle
import dotenv
import requests
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL endpoint API FastAPI lokal
API_URL = "http://127.0.0.1:8000/api/chat"

# Fungsi untuk menangani perintah /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    welcome_message = """
🤖 Selamat datang di Chatbot UMKM & Data Penduduk!

Saya dapat membantu Anda dengan:
- Informasi salam dan ucapan
- berbagai data dan informasi umum

Ketik pesan Anda untuk memulai percakapan!
"""
    await update.message.reply_text(welcome_message)

# Fungsi untuk mengirim pesan pengguna ke API dan menerima respons
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mengirim pesan teks dari pengguna ke API chatbot dan membalas dengan respons."""
    user_text = update.message.text
    response_text = await get_chatbot_response(user_text)
    await update.message.reply_text(response_text)

async def get_chatbot_response(user_text):
    """Menerima respons dari API chatbot."""
    try:
        logger.info(f"Mengirim '{user_text}' ke API...")
        response = requests.post(API_URL, json={"text": user_text}, timeout=10)
        response.raise_for_status()  # Memunculkan HTTPError jika respons buruk
        
        result = response.json()
        
        # Mengambil respons dari JSON yang dikembalikan oleh FastAPI
        response_data = result.get("response", "Maaf, saya tidak bisa memberikan respons yang tepat.")
        
        logger.info(f"Respons API: {response_data}")
        return response_data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error saat terhubung ke API: {e}")
        return "Maaf, sistem chatbot sedang tidak tersedia. Coba lagi nanti."
    except Exception as e:
        logger.error(f"Terjadi kesalahan tak terduga: {e}")
        return "Maaf, terjadi kesalahan tak terduga."
    
def main():
    # Ganti dengan token bot Telegram Anda
    dotenv.load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    # Buat instance aplikasi bot
    app = ApplicationBuilder().token(TOKEN).build()

    # Tambahkan handler untuk perintah dan pesan teks
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Jalankan bot
    print("Bot Telegram berjalan. Tekan Ctrl+C untuk berhenti.")
    app.run_polling()

if __name__ == '__main__':
    main()
