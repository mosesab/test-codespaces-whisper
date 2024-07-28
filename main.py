import sys
import os
import logging
import time
import boto3.exceptions
import requests
import boto3
import traceback
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from urllib.parse import urlparse
from whisper_backend import MyWhisper
####### ----   REGION - Load Bot Config Variables   ----   ####### 
logger = logging.getLogger(__name__)

def output_transcript(o, transcriptfile, start, now=None):
    try:
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from startinning of processing, in milliseconds
        #    - start and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print(f"Start:{o[0]:1.0f} ; End:{o[1]:1.0f} ; Text: {o[2]}",file=transcriptfile,flush=True)
            print(f"Start:{o[0]:1.0f} ; End:{o[1]:1.0f} ; Text: {o[2]}",flush=True)
        else:
            pass # No text, so no output
    except Exception as e:
        traceback.print_exc()
        print(f"output_transcript has an error: {e}")


def get_file_path_from_url(audio_file_url, retry_attempts=0):
    retry_attempts += 1
    try:
        parsed_url = urlparse(audio_file_url)
        print(f"audio url: {audio_file_url}")
        env_file_extension = os.getenv("AUDIO_FILE_EXTENSION")
        # validate the audio file
        if 's3' in parsed_url.netloc:
            try:
                # Download the file from S3
                s3 = boto3.client('s3')
                # Extract the bucket name and the object key from the URL
                bucket_name = parsed_url.netloc.split('.')[0]
                object_key = parsed_url.path.lstrip('/')
                # Get the file name and extension from the object key
                file_name, file_extension = os.path.splitext(os.path.basename(object_key))
                if env_file_extension !="":
                    file_extension = '.' + str(env_file_extension)
                file_path = os.path.join(os.getcwd(), file_name + file_extension)
                s3.download_file(bucket_name, object_key, file_path)
                return file_path
            except Exception as e:
                traceback.print_exc()
                print(f"Error downloading the file from s3 bucket: {e}")
                return None
        else:
            # Get the file name from the URL
            file_name = os.path.basename(urlparse(audio_file_url).path)
            file_path = os.path.join(os.getcwd(), file_name)
            print(f"file_name: {file_name}")
            # Download the file from the URL
            try:
                response = requests.get(audio_file_url)
                response.raise_for_status()  # Check for HTTP errors
                # Save the file to the current working directory
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"File downloaded to {file_path}")
                return file_path
            except requests.exceptions.RequestException as e:
                traceback.print_exc()
                print(f"Http Error downloading the file: {e}")
                return None
    except Exception as e:
        # retry initialize 5 times
        if retry_attempts >= 5:
            traceback.print_exc()
            print(f"Error downloading the file: {e}")
            return None
        else:
            print(f"An ERROR OCCURED: While Trying To get_file_path_from_url, Retrying {retry_attempts}.")
            return get_file_path_from_url(audio_file_url, retry_attempts)
        








def send_transcript_to_S3(transcript_file_path, retry_attempts=0):
    # Upload the recording file to the S3 bucket
    retry_attempts += 1
    try:
        user_id = os.getenv("USER_ID")
        meeting_id = os.getenv("MEETING_ID") 
        time_stamp = str(time.strftime("%Y-%m-%d-%H-%M-%S"))
        aws_access_key_id = os.getenv['AWS_ACCESS_KEY_ID']
        aws_secret_access_key = os.getenv['AWS_SECRET_ACCESS_KEY']
        object_name = f"{user_id}+{meeting_id}+{time_stamp}"
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        # Rename the file
        file_name, extension = os.path.splitext(transcript_file_path)
        object_name = object_name + extension
        # Initialize S3 client
        if (len(aws_access_key_id) > 1) or (len(aws_secret_access_key) > 1):
            boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        else:
            s3_client = boto3.client('s3')
            
        if object_name is None:
            object_name = file_name + str(time_stamp)

        response = s3_client.upload_file(
            transcript_file_path, 
            bucket_name, 
            object_name,
            ExtraArgs={
                'ServerSideEncryption': 'aws:kms',
                'BucketKeyEnabled': True
            }
        )
        print(f"Transcript File {file_name} uploaded successfully to {bucket_name}/{object_name}")
        return response
    except FileNotFoundError:
        traceback.print_exc()
        print("send_transcript_to_S3: Error, The file was not found")
        return None
    except NoCredentialsError:
        traceback.print_exc()
        print("send_transcript_to_S3: Error, Credentials not available")
        return None
    except PartialCredentialsError:
        traceback.print_exc()
        print("send_transcript_to_S3: Error, Incomplete credentials provided")
        return None
    except Exception as e:
        # retry initialize 5 times
        if retry_attempts >= 5:
            traceback.print_exc()
            print(f"send_transcript_to_S3: Error uploading file: {e}")
            return None
        else:
            print(f"An ERROR OCCURED: While Trying To send_transcript_to_S3, Retrying {retry_attempts}.")
            return send_transcript_to_S3(transcript_file_path, retry_attempts)
        
        


def process_audio_with_whisper():
    try:
        # Load Variables
        audio_file_url = os.getenv("AUDIO_FILE_URL") # MEETING_LINK can also be a filepath to an audio file
        audio_file_path = get_file_path_from_url(audio_file_url)
        if audio_file_path is None:
            return
        model = os.getenv("WHISPER_MODEL") 
        system_processor = os.getenv("SYS_PROCESSOR")
        model_dir = os.getenv("WHISPER_MODEL_DIR")
        if (model_dir =="") or (model_dir.lower() == "None".lower()):
            model_dir = None
        elif model_dir =="whisper_small_model":
            model_dir = os.path.join(os.getcwd(), model_dir)
        min_chunk = os.getenv("MIN_AUDIO_CHUNCK")
        min_chunk = int(min_chunk)
        language = "auto"
        task = "transcribe"
        use_vad = os.getenv("USE_VAD")
        use_aws_s3 = os.getenv("USE_AWS_S3")
        buffer_trimming_sec = 15
        log_level = os.getenv("LOG_LEVEL")
        transcript_file_path = "transcript.txt"
        model_cache_dir = None 
        logfile = sys.stderr
        my_whisper = MyWhisper()
        logging.basicConfig(#format='%(name)s 
                format='%(levelname)s\t%(message)s')
        logger.setLevel(log_level)
        logging.getLogger("whisper_online").setLevel(log_level)
        processor = 0
        if system_processor == "cuda-float16":
            processor = 1
        elif system_processor == "cuda-int8":
            processor = 2
        
        asr, asr_processor = my_whisper.create_asr_processor(model, model_cache_dir, model_dir, processor, task, use_vad, buffer_trimming_sec, lan=language, logfile=logfile)

        # load the audio into the LRU cache before we start the timer
        a = my_whisper.load_audio_chunk(os.path.join(os.getcwd(), "warm_up_whisper.wav"),0,1)
        # warm up the ASR because the very first transcribe takes much more time than the other
        try:
            asr.transcribe(a)
        except Exception as e:
            print(f"Warning: ASR Warm-Up Warning: {e}")
        start = 0.0
        """
        This mode processes the audio in chunks, which is useful when dealing with longer audio files where processing the entire file at once is impractical.
        """
        duration = len(my_whisper.load_audio(audio_file_path))/16000
        logger.info("Audio duration is: %2.2f seconds" % duration)
        end = start + min_chunk
        print(f"Starting Transcribtion of Speaker Audio File")
        with open(transcript_file_path, "w") as transcriptfile:
            while True:
                a = my_whisper.load_audio_chunk(audio_file_path,start,end)
                print(f"Processing Audio from start= {start} seconds to end= {end} seconds")
                asr_processor.insert_audio_chunk(a)
                try:
                    o = asr_processor.process_iter()
                except AssertionError as e:
                    traceback.print_exc()
                    logger.error(f"assertion error: {repr(e)}")
                    print(f"assertion error: {repr(e)}")
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Processing Error: {str(e)}")
                    print(f"Processing Error: {str(e)}")
                else:
                    output_transcript(o, transcriptfile, start, now=end)
                if end >= duration:
                    break
                start = end
                if end + min_chunk > duration:
                    end = duration
                else:
                    end += min_chunk
            now = duration
        o = asr_processor.finish()
        with open(transcript_file_path, "w") as transcriptfile:
            output_transcript(o, transcriptfile, start, now=now)
        if use_aws_s3:
            send_transcript_to_S3(transcript_file_path)
    except Exception as e:
        traceback.print_exc()
        print(f"process_audio_with_whisper has an error: {e}")


if __name__ == '__main__':
    process_audio_with_whisper()


