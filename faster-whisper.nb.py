# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
os.environ["CT2_USE_MKL"] = "0"
os.environ["CT2_VERBOSE"] = "1"

# %%
from faster_whisper import WhisperModel

model_size = "small.en"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", cpu_threads=4)

# %%
import time

start = time.time()
segments, info = model.transcribe("GovTech_MFTMEP1_16000_mono_16.wav", beam_size=5, language='en')

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print(f'Time: {time.time() - start}')

# %%
os.environ['LD_LIBRARY_PATH']

# %%
# %load_ext line_profiler

# %%
from stream import WhisperModel
import logging, sys

model_size = "medium.en"

# Run on GPU with FP16
# model = WhisperModel(model_size, compute_type="int8", device="cpu", cpu_threads=0)
model = WhisperModel(model_size, download_root='/home/chiiyeh/dev/ct_model')
consoleHandler = logging.StreamHandler(stream=sys.stdout)
consoleHandler.setLevel(logging.INFO)
model.logger.addHandler(consoleHandler)
model.logger.setLevel(logging.INFO)

# %%
# !$PWD

# %%
tokenizer, options, stream_options = model.init_options(
    beam_size=1,
    best_of=1,
    condition_on_previous_text=True, 
    max_prompt_tokens=100,
    drop_dup_prompt=True,
    is_prompt_sentence=True,
    finalised_segment_gap=3,
    rm_seconds=0.3, 
    word_timestamps=True, 
    without_timestamps=True, 
    # log_prob_threshold=-2, 
    max_tokens_per_second=5,
    use_prefix=False,
    prefix_drop_num_tokens=4,
    drop_prefix_duration=0,
    )
# model.warm_start(tokenizer, options, stream_options)

# %%
model.warm_start(tokenizer, options, stream_options)

# %%
# %lprun -f model.simulate_streaming -f model.generate_segments model.simulate_streaming("GovTech_MFTMEP1_16000_mono_16.wav", tokenizer, options, stream_options, min_interval=3)

# %%
tokenizer.no_timestamps

# %%
tokenizer.timestamp_begin

# %%
tokenizer.sot

# %%
tokenizer.sot_prev

# %%
tokenizer, options, stream_options = model.init_options(
    beam_size=1,
    best_of=3,
    condition_on_previous_text=True, 
    no_repeat_ngram_size=3,
    max_prompt_tokens=100,
    drop_dup_prompt=True,
    is_prompt_sentence=True,
    finalised_segment_gap=3,
    rm_seconds=0.3, 
    word_timestamps=True, 
    without_timestamps=True, 
    # log_prob_threshold=-2, 
    max_tokens_per_second=5,
    use_prefix=False,
    prefix_drop_num_tokens=4,
    drop_prefix_duration=0,
    initial_prompt="This is the start of the conversation."
    )
# model.max_length=448
# model.warm_start(tokenizer, options, stream_options)

# %%
# %lprun -f model.simulate_fixed_interval_streaming -f model.generate_segments model.simulate_fixed_interval_streaming("GovTech_MFTMEP1_16000_mono_16.wav", tokenizer, options, stream_options, interval=7)

# %%
round(1.4)

# %%
stream_options

# %%
tokenizer

# %%
import os
import soundfile as sf
import time
import numpy as np
import importlib
import jeff_live
importlib.reload(jeff_live)
from jeff_live import WhisperModel_live

model_size = 'small.en'
model_dir = "pretrained_models/faster_whisper/"
# audio_file = "data/wav/IMDA_test_60s.wav"
audio_file = "GovTech_MFTMEP1_16000_mono_16.wav"

# Load model before model files have been downloaded
# faster_model_live = WhisperModel_live(model_size, compute_type="int8", download_root=model_dir)
# Load model after model files have been downloaded
faster_model_live = WhisperModel_live(model_size, compute_type="int8")

def run_audio(audio_file, faster_model_live):
  wav, fs = sf.read(audio_file)
  wav_len = len(wav)
  
  RATE = 16000
  CHUNK_S = 7
  CHUNK = CHUNK_S * RATE
  # max_time = 440.0
  max_time = -1
  compression_ratio_threshold = 2.4
  
  if max_time > 0:
      wav_len = min(wav_len, int(max_time*RATE))
  chunk_starts = list(range(0, wav_len, CHUNK))
  final_text = ''
  seek = 0.0
  for chunk_start in chunk_starts[0:]:
      t1 = time.time()
      non_final_text = ''
      repeat_text = ''
      start_idx = min(chunk_start, int(seek*RATE))
      end_idx = min(wav_len, chunk_start + CHUNK)
  
      # Make sure audio chunk is less than 30s
      # Skip to most recent 30s if longer
      start_idx = max(start_idx, end_idx - 30*RATE)
      start_time = start_idx/RATE
      end_time = end_idx/RATE
  
      print(f'Transcribing {start_time:.2f}->{end_time:.2f}s...')
      wav_chunk = np.array(wav[start_idx:end_idx], dtype=np.float32)
      
      segments_live, info = faster_model_live.transcribe(wav_chunk, beam_size=1, temperature=0, vad_filter=False, word_timestamps=False)
      seek = start_time
      for seg in segments_live:
          # print(seg)
          print(f'SEG {start_time+seg.start:.2f}->{start_time+seg.end:.2f}, seek:{start_time+seg.seek_live:.2f}'
                  + f', final:{seg.final}, compression_ratio: {seg.compression_ratio:.2f}, words: {seg.text.lstrip()}')
  
          if seg.compression_ratio > compression_ratio_threshold:
              repeat_text += seg.text
  
          elif seg.final:
              final_text += seg.text
              seek = start_time + seg.seek_live
          else:
              non_final_text = seg.text
  
      t2 = time.time()
      time_elapsed = t2 - t1
      print(f'Output: {final_text.lstrip()} | {non_final_text.lstrip()}')
      if repeat_text:
          print(f'REPEAT: {repeat_text.lstrip()}')
      print(f'Time elapsed: {time_elapsed:0.2f}s\n')

# %%
# %lprun -f run_audio run_audio(audio_file, faster_model_live)

# %%
