# Train the model
# Usage: bash train.sh
echo "perbaiki file csv mentah"
python scripts/fix_csv.py

echo "perbaiki format csv yang rusak"
python scripts/fix_csv_malformed.py

echo "pisah data latih"
python scripts/split_data.py

echo "latih model"
python chatbot_training.py

echo "model berhasil dilatih dan tersimpan di direktori model"