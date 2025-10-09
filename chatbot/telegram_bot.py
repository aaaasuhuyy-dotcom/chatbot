# Python 3.9+
import os
import pickle
import dotenv
import requests
import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Konfigurasi
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_BASE_URL = "http://127.0.0.1:8000"  # FastAPI local server

# Fungsi untuk menangani perintah /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    welcome_message = """
🤖 **Selamat datang di Chatbot UMKM & Data Penduduk!**

Saya dapat membantu Anda dengan:
• 📊 Informasi umum
• 🏢 Informasi UMKM dan bisnis
• 🕐 Jam operasional instansi
• ℹ️ informasi prosedur pembuatan kk, nib, ak1

**Perintah yang tersedia:**
/start - Memulai bot
/help - Menampilkan bantuan
/stats - Status sistem
/intents - Daftar intent yang tersedia
/debug - Mode debug prediksi

Ketik pesan Anda atau gunakan tombol cepat di bawah!
"""
    
    # Create keyboard buttons
    keyboard = [
        [KeyboardButton("cara buat kk"), KeyboardButton("cara buat nib")],
        [KeyboardButton("cara buat ak1")]
        # KeyboardButton("Dimana lokasinya?")],
        # [KeyboardButton("Informasi UMKM"), KeyboardButton("Data penduduk")],
        # [KeyboardButton("Bantuan"), KeyboardButton("Fitur apa saja?")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        welcome_message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

# Fungsi untuk menangani perintah /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /help"""
    help_text = """
🆘 **Bantuan Penggunaan Bot**

**Cara menggunakan:**
1. Ketik pertanyaan langsung
2. Gunakan tombol cepat untuk pertanyaan umum
3. Bot akan memahami maksud Anda secara otomatis

**Contoh pertanyaan:**
• "Kapan Bappenda buka?"
• "Apa itu nib?"
• "Bagaimana cara daftar nib?"

**Perintah yang tersedia:**
/start - Memulai bot
/help - Menampilkan bantuan
/stats - Status sistem
/intents - Daftar intent yang tersedia
/debug - Mode debug prediksi

**Teknologi:**
• LSTM Neural Network
• IndoBERT Transformer
• Hybrid AI System
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

# Fungsi untuk menangani perintah /stats
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /stats - menampilkan status sistem"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats_text = f"""
📊 **Status Sistem**

✅ **Status:** {data.get('status', 'Unknown')}
🤖 **Model LSTM:** {'🟢 Loaded' if data.get('model_loaded') else '🔴 Not Loaded'}
🧠 **Model BERT:** {'🟢 Available' if data.get('bert_available') else '🔴 Not Available'}
📁 **Jumlah Intent:** {data.get('intents_count', 0)}
🗂️ **Total Patterns:** {data.get('total_patterns', 0)}

**Sistem Hybrid AI aktif dan siap melayani!**
"""
        else:
            stats_text = "❌ **Gagal mengambil status sistem**"
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        stats_text = "❌ **Tidak dapat terhubung ke server AI**"
    
    await update.message.reply_text(stats_text, parse_mode='Markdown')

# Fungsi untuk menangani perintah /intents
async def intents_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /intents - menampilkan daftar intent"""
    try:
        response = requests.get(f"{API_BASE_URL}/intents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            intents = data.get('available_intents', [])
            details = data.get('intents_details', [])
            
            intents_text = "📋 **Daftar Intent yang Tersedia**\n\n"
            
            for detail in details:
                intent_name = detail.get('intent', 'Unknown')
                response_type = detail.get('response_type', 'Unknown')
                patterns_count = detail.get('patterns_count', 0)
                responses_count = detail.get('responses_count', 0)
                
                intents_text += f"• **{intent_name}**\n"
                intents_text += f"  Type: {response_type} | "
                intents_text += f"Patterns: {patterns_count} | "
                intents_text += f"Responses: {responses_count}\n\n"
            
            intents_text += f"📊 **Total: {len(intents)} intent**"
        else:
            intents_text = "❌ **Gagal mengambil daftar intent**"
    except Exception as e:
        logger.error(f"Error getting intents: {e}")
        intents_text = "❌ **Tidak dapat terhubung ke server AI**"
    
    await update.message.reply_text(intents_text, parse_mode='Markdown')

# Fungsi untuk menangani perintah /debug
async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /debug - mode debug prediksi"""
    if not context.args:
        debug_text = """
🔧 **Mode Debug Prediksi**

Gunakan perintah:
`/debug <teks_pertanyaan>`

Contoh:
`/debug kapan bappenda buka?`
`/debug informasi umkm`

Bot akan menampilkan detail prediksi dari sistem AI.
"""
        await update.message.reply_text(debug_text, parse_mode='Markdown')
        return
    
    text_to_debug = ' '.join(context.args)
    await send_debug_prediction(update, text_to_debug)

# Fungsi untuk mengirim pesan pengguna ke API dan menerima respons
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mengirim pesan teks dari pengguna ke API chatbot dan membalas dengan respons."""
    user_text = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    logger.info(f"📩 Message from {username} ({user_id}): '{user_text}'")
    
    # Kirim typing action
    await update.message.chat.send_action(action="typing")
    
    # Dapatkan respons dari API
    response_data = await get_chatbot_response(user_text)
    
    # Kirim respons ke pengguna
    await update.message.reply_text(response_data)
    
    logger.info(f"📤 Response to {username}: '{response_data[:100]}...'")

async def get_chatbot_response(user_text: str) -> str:
    """Menerima respons dari API chatbot."""
    try:
        logger.info(f"🔗 Sending to API: '{user_text}'")
        response = requests.post(
            f"{API_BASE_URL}/api/chat", 
            json={"text": user_text}, 
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Format response dengan informasi tambahan jika tersedia
        response_text = result.get("response", "Maaf, saya tidak bisa memberikan respons yang tepat.")
        intent = result.get("predicted_intent", "unknown")
        confidence = result.get("confidence", 0)
        method = result.get("method_used", "unknown")
        
        # Tambahkan info kecil di footer jika confidence rendah
        if confidence < 0.6:
            response_text += f"\n\n⚡ *[AI Confidence: {confidence:.1%}]*"
        
        logger.info(f"✅ API Response - Intent: {intent}, Confidence: {confidence:.3f}, Method: {method}")
        return response_text
    
    except requests.exceptions.ConnectionError:
        logger.error("❌ ConnectionError: Cannot connect to API server")
        return "🔌 **Server AI sedang offline**\n\nMohon tunggu sebentar atau coba lagi nanti."
    
    except requests.exceptions.Timeout:
        logger.error("❌ Timeout: API request timed out")
        return "⏰ **Waktu permintaan habis**\n\nSilakan coba lagi dengan pertanyaan yang lebih singkat."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ RequestException: {e}")
        return "❌ **Terjadi kesalahan koneksi**\n\nMohon coba lagi dalam beberapa saat."
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return "🤖 **Terjadi kesalahan tak terduga**\n\nTim technical sudah diberitahu. Silakan coba lagi nanti."

async def send_debug_prediction(update: Update, text: str):
    """Mengirim informasi debug prediksi"""
    try:
        logger.info(f"🔧 Debug request for: '{text}'")
        response = requests.get(
            f"{API_BASE_URL}/api/debug-prediction",
            params={"text": text},
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        debug_text = f"""
🔧 **Debug Prediction Results**

📝 **Input:** `{data.get('input_text', 'N/A')}`

🤖 **LSTM Prediction:**
• Intent: `{data['lstm_prediction'].get('intent', 'N/A')}`
• Confidence: `{data['lstm_prediction'].get('confidence', 0):.3f}`
• Status: `{data['lstm_prediction'].get('status', 'N/A')}`

🧠 **BERT Prediction:**
• Intent: `{data['bert_prediction'].get('intent', 'N/A')}`
• Confidence: `{data['bert_prediction'].get('confidence', 0):.3f}`
• Status: `{data['bert_prediction'].get('status', 'N/A')}`

🎯 **Final Prediction:**
• Intent: `{data['fused_prediction'].get('intent', 'N/A')}`
• Confidence: `{data['fused_prediction'].get('confidence', 0):.3f}`
• Method: `{data['fused_prediction'].get('method', 'N/A')}`
• Sources: `{data['fused_prediction'].get('sources', [])}`

💬 **Response:** {data.get('final_response', 'N/A')}
"""
        await update.message.reply_text(debug_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Debug prediction error: {e}")
        await update.message.reply_text("❌ **Gagal mendapatkan info debug**")

# Fungsi untuk menangani error
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk error yang tidak terduga"""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # Kirim pesan error ke pengguna
    if update and update.effective_message:
        error_text = """
❌ **Terjadi kesalahan sistem**

Mohon maaf atas ketidaknyamanannya. Silakan coba lagi dalam beberapa saat.
"""
        await update.effective_message.reply_text(error_text)

def main():
    """Main function untuk menjalankan bot"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN tidak ditemukan!")
        print("Pastikan Anda telah mengatur TELEGRAM_BOT_TOKEN di file .env")
        return
    
    try:
        # Buat instance aplikasi bot
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Tambahkan handler untuk commands
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("stats", stats_command))
        app.add_handler(CommandHandler("intents", intents_command))
        app.add_handler(CommandHandler("debug", debug_command))
        
        # Tambahkan handler untuk pesan teks
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Tambahkan error handler
        app.add_error_handler(error_handler)
        
        # Jalankan bot
        print("🤖 Telegram Bot berhasil dijalankan!")
        print("📍 Listening for messages...")
        print("⏹️  Press Ctrl+C to stop")
        
        app.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"❌ Gagal menjalankan bot: {e}")

if __name__ == '__main__':
    main()