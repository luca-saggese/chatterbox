#!/bin/bash
# Uso: ./smart_cut.sh input.mp3 00:05:30 00:35:30
# Trova le pause più vicine a inizio/fine e taglia di conseguenza,
# analizzando solo 10 minuti attorno a ciascun punto (non tutto il file).

if [ "$#" -ne 3 ]; then
  echo "Uso: $0 <input.mp3> <start_time> <end_time>"
  exit 1
fi

INPUT="$1"
START="$2"
END="$3"

TMP_START_LOG="silence_start.log"
TMP_END_LOG="silence_end.log"

# Converti HH:MM:SS in secondi
to_seconds() {
  IFS=: read -r h m s <<< "$1"
  echo "$((10#$h * 3600 + 10#$m * 60 + 10#${s%.*}))"
}

start_sec=$(to_seconds "$START")
end_sec=$(to_seconds "$END")

# === 1️⃣ Analizza 10 minuti dopo lo start ===
echo "🔎 Analizzo 10 minuti dopo $START per trovare la prima pausa..."
ffmpeg -hide_banner -loglevel info \
  -ss "$START" -t 600 -i "$INPUT" \
  -af silencedetect=noise=-30dB:d=0.5 -f null - 2> "$TMP_START_LOG"

first_silence=$(grep "silence_end" "$TMP_START_LOG" | head -n 1 | awk '{print $5}')
if [ -z "$first_silence" ]; then
  echo "⚠️ Nessuna pausa trovata vicino allo start, userò l'inizio esatto."
  first_silence=0
fi
real_start=$(echo "$start_sec + $first_silence" | bc)

# === 2️⃣ Analizza 10 minuti prima dell’end ===
echo "🔎 Analizzo 10 minuti prima di $END per trovare l'ultima pausa..."
search_start=$(echo "$end_sec - 600" | bc)
if (( $(echo "$search_start < 0" | bc -l) )); then search_start=0; fi

ffmpeg -hide_banner -loglevel info \
  -ss "$search_start" -t 600 -i "$INPUT" \
  -af silencedetect=noise=-30dB:d=0.5 -f null - 2> "$TMP_END_LOG"

last_silence=$(grep "silence_start" "$TMP_END_LOG" | tail -n 1 | awk '{print $5}')
if [ -z "$last_silence" ]; then
  echo "⚠️ Nessuna pausa trovata vicino alla fine, userò l'end esatto."
  real_end="$end_sec"
else
  real_end=$(echo "$search_start + $last_silence" | bc)
fi

# === 3️⃣ Taglia il file ===
# Extract directory and filename separately to build correct output path
input_dir=$(dirname "$INPUT")
input_name=$(basename "$INPUT" .mp3)
output="${input_dir}/cut_${input_name}.mp3"
echo "✂️  Taglio da ${real_start}s a ${real_end}s → $output"
ffmpeg -hide_banner -loglevel error -i "$INPUT" -ss "$real_start" -to "$real_end" -c copy "$output"

# === 4️⃣ Pulisci ===
rm -f "$TMP_START_LOG" "$TMP_END_LOG"
echo "✅ Fatto! File creato: $output"
