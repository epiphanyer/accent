curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "text=Hello world" \
     -F "audio=@../output.mp3"