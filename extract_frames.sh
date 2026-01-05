#!/bin/bash
# Usage: ./extract_frames_from_list.sh "<folder>" <frames.txt>

set -euo pipefail

FOLDERNAME="$1"
LISTFILE="$2"

OUTDIR="${FOLDERNAME}/frames_extracted"
mkdir -p "$OUTDIR"

# Read raw lines safely; handle no trailing newline + strip CRs
while IFS= read -r LINE || [ -n "$LINE" ]; do
  # Trim CR and whitespace
  LINE="${LINE//$'\r'/}"
  LINE="${LINE#"${LINE%%[![:space:]]*}"}"
  LINE="${LINE%"${LINE##*[![:space:]]}"}"

  # Skip blank/comment
  [[ -z "$LINE" || "$LINE" =~ ^# ]] && continue

  # Split "file,timestamp" without relying on IFS field-splitting
  FILENAME="${LINE%%,*}"
  TIMESTAMP="${LINE#*,}"

  # Safety: auto-fix if first 'r' got dropped somehow
  [[ "$FILENAME" == gb_* ]] && FILENAME="r${FILENAME}"

  INPUT_PATH="${FOLDERNAME}/${FILENAME}"
  if [ ! -f "$INPUT_PATH" ]; then
    echo "❌ File not found: $INPUT_PATH"
    continue
  fi

  BASENAME="${FILENAME%.mp4}"
  CLEAN_TIMESTAMP="${TIMESTAMP//:/-}"
  OUTFILE="${OUTDIR}/${BASENAME}_${CLEAN_TIMESTAMP}.jpg"

  echo "🎞 Extracting frame from $FILENAME at $TIMESTAMP → $OUTFILE"
  ffmpeg -ss "$TIMESTAMP" -i "$INPUT_PATH" -frames:v 1 -q:v 2 "$OUTFILE" -y >/dev/null 2>&1

  if [ -f "$OUTFILE" ]; then
    echo "✅ Saved: $OUTFILE"
  else
    echo "⚠️ Failed to extract frame for $FILENAME at $TIMESTAMP"
  fi
done < "$LISTFILE"

echo "✨ Done. Frames saved to: $OUTDIR"
