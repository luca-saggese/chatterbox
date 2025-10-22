#!/bin/bash
docker build -t chatterbox .
echo "✅ Build completata!"
echo "👉 Per eseguire il container usa:"
echo "docker run --rm -it --gpus all -p 8085:8080 --name chatterbox -v /home/lvx/huggingface:/huggingface chatterbox"

