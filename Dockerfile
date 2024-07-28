# Use as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .


# Set any required environment variables 
ENV AUDIO_FILE_URL="s3bucket/recordings/001.wav"
ENV AUDIO_FILE_EXTENSION=""
ENV AWS_BUCKET_NAME="transcripts"
ENV USER_ID="DEFAULT_USER"
ENV MEETING_ID="DEFAULT_FILE_ID"
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""

ENV WHISPER_MODEL="small"
ENV WHISPER_MODEL_DIR="whisper_small_model"
ENV SYS_PROCESSOR="cpu"
ENV MIN_AUDIO_CHUNCK=10
ENV USE_VAD=True
ENV USE_AWS_S3=True
ENV LOG_LEVEL="INFO"


# Define the command to run the main.py script
CMD ["python", "main.py"]


