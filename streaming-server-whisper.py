#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Wei Kang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A server for streaming ASR recognition. By streaming it means the audio samples
are coming in real-time. You don't need to wait until all audio samples are
captured before sending them for recognition.

It supports multiple clients sending at the same time.

Usage:
    ./streaming_server.py --help

    ./streaming_server.py

Please refer to
https://k2-fsa.github.io/sherpa/python/streaming_asr/conformer/index.html
for details
"""

import argparse
import asyncio
import http
import json
import logging
import socket
import ssl
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import websockets
from stream import WhisperModel



def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
    )

    parser.add_argument(
        "--whisper-model-filename",
        type=str,
        required=True,
        help="""Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, medium, medium.en, large-v1, or large-v2) or a path to a converted
            model directory. When a size is configured, the converted model is downloaded
            from the Hugging Face Hub.
        """,
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "fp16", "fp32"],
        help="Compute type.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Number of cpu threads to use",
    )

    parser.add_argument(
        "--cpu-thread",
        type=int,
        default=0,
        help="Number of cpu threads to use",
    )

    parser.add_argument(
        "--min-new-audio-dur",
        type=float,
        default=3,
        help="Minimum newly uploaded audio before starting to process.",
    )

    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=1,
        help="Number of threads for NN computation and decoding.",
    )

    parser.add_argument(
        "--max-message-size",
        type=int,
        default=(1 << 20),
        help="""Max message size in bytes.
        The max size per message cannot exceed this limit.
        """,
    )

    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=200,
        help="Max number of messages in the queue for each connection.",
    )

    parser.add_argument(
        "--max-active-connections",
        type=int,
        default=2,
        help="""Maximum number of active connections. The server will refuse
        to accept new connections once the current number of active connections
        equals to this limit.
        """,
    )

    parser.add_argument(
        "--certificate",
        type=str,
        help="""Path to the X.509 certificate. You need it only if you want to
        use a secure websocket connection, i.e., use wss:// instead of ws://.
        You can use sherpa/bin/web/generate-certificate.py
        to generate the certificate `cert.pem`.
        """,
    )

    return parser.parse_args()


class StreamingServer(object):
    def __init__(
        self,
        whisper_model_filename: str,
        compute_type: str,
        device: str,
        cpu_thread: int,
        min_new_audio_dur: float,
        nn_pool_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        certificate: Optional[str] = None,
    ):
        """
        Args:
          whisper_model_filename:
            Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, medium, medium.en, large-v1, or large-v2) or a path to a converted
            model directory. When a size is configured, the converted model is downloaded
            from the Hugging Face Hub.
          compute_type:
            The compute type one of ["int8", "fp16", "fp32"]
          device:
            device to use either "cpu" or "cuda"
          cpu_thread:
            number of cpu threads to use. 0 will use all.
          min_new_audio_dur:
            Minimum unprocessed audio duration to run.
          nn_pool_size:
            Number of threads for the thread pool that is responsible for
            neural network computation and decoding.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./sherpa/bin/web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """

        self.model = WhisperModel(
            whisper_model_filename,
            compute_type=compute_type,
            device=device,
            cpu_threads=cpu_thread,
        )

        self.min_new_audio_dur = min_new_audio_dur
        #! Supposed to be set again when the speech comes in.
        self.tokenizer, self.options, self.stream_options = self.model.init_options(
            beam_size=1,
            condition_on_previous_text=False,
            finalised_segment_gap=2,
            word_timestamps=True,
            without_timestamps=True,
            drop_out_of_bound=False,
            log_prob_threshold=None,
            use_prefix=False,
            prefix_drop_num_tokens = 5,
            temperature=0,
        )

        self.certificate = certificate

        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

    async def warmup(self) -> None:
        """Do warmup to the torchscript model to decrease the waiting time
        of the first request.

        See https://github.com/k2-fsa/sherpa/pull/100 for details
        """
        logging.info("Warmup start")
        await asyncio.get_running_loop().run_in_executor(self.nn_pool, self.model.warm_start, self.tokenizer, self.options, self.stream_options)

        logging.info("Warmup done")

    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if self.current_active_connections < self.max_active_connections:
            self.current_active_connections += 1
            return None

        # Refuse new connections
        status = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."

        return status, header, response

    async def run(self, port: int):
        # task = asyncio.create_task(self.stream_consumer_task())
        await self.warmup()
        if self.certificate:
            logging.info("Using certificate: %s", self.certificate)
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.certificate)
        else:
            ssl_context = None
            logging.info("No certificate provided")

        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
            ssl=ssl_context,
            ping_timeout=None
        ):
            ip_list = ["0.0.0.0", "localhost", "127.0.0.1"]
            ip_list.append(socket.gethostbyname(socket.gethostname()))
            proto = "http://" if ssl_context is None else "https://"
            s = "Please visit one of the following addresses:\n\n"
            for p in ip_list:
                s += "  " + proto + p + f":{port}" "\n"
            logging.info(s)

            await asyncio.Future()  # run forever

        # await task  # not reachable

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            await self.handle_connection_impl(socket)
        except websockets.exceptions.ConnectionClosedError:
            logging.info("%s disconnected", socket.remote_address)
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1

            logging.info(
                f"Disconnected: {socket.remote_address}. "
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )

    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )

        sampling_rate = 16000
        IsSending=True
        current_audio = None
        idx = 0
        seek = 0
        previous_tokens = []
        prefix_tokens = []
        initial_prompt_tokens = []
        current_audio = None
        is_fast_load = True
        received_audio = []
        not_finalised_messages = []
        temperature_idx = 0
        while IsSending:
            received_audio = await self.recv_audio_samples(socket)
            if received_audio is None:
                break
            elif isinstance(received_audio,str):
                continue
            while is_fast_load or (len(received_audio) < self.min_new_audio_dur*sampling_rate):
                test1 = time.perf_counter()
                samples = await self.recv_audio_samples(socket)
                duration_wait = time.perf_counter() - test1
                if samples is None:
                    IsSending=False
                    break
                received_audio = np.concatenate([received_audio, samples])
                is_fast_load = duration_wait < len(samples)/(2*sampling_rate)
            is_fast_load = True
            logging.debug("%s: Added Duration: %s", socket.remote_address, f"{len(received_audio)/sampling_rate:.2f}")
            if current_audio is not None:
                current_audio = np.concatenate([current_audio, received_audio])
            else:
                current_audio = received_audio

            # TODO(fangjun): At present, we assume the sampling rate
            # of the received audio samples is always 16000.
            time_start = time.perf_counter()
            (
                output_segments,
                current_audio,
                idx,
                seek,
                temperature_idx,
                previous_tokens,
                prefix_tokens,
            ) = await asyncio.get_running_loop().run_in_executor(
                self.nn_pool, 
                self.model.generate_segments,
                current_audio,
                self.tokenizer,
                self.options,
                self.stream_options,
                idx,
                seek,
                temperature_idx,
                previous_tokens,
                prefix_tokens,
                initial_prompt_tokens,
            )
            if output_segments:
                not_finalised_messages = []
            for segment in output_segments:
                message = segment._asdict()
                if message["words"]:
                    message["words"] = [word._asdict() for word in segment.words]
                else:
                    del message["words"]
                # print(segment._asdict())
                message["final"] = True or not IsSending
                if not message["final"]:
                    not_finalised_messages.append(message)
                await socket.send(json.dumps(message))
            logging.debug('%s: Processing time: %s', socket.remote_address, f"{time.perf_counter()-time_start:.2f}")
        for message in not_finalised_messages:
            message["final"] = True
            # print(segment._asdict())
            await socket.send(json.dumps(message)) 

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[Union[np.ndarray, str]]:
        """Receives a tensor from the client.

        Each message contains either a bytes buffer containing audio samples
        in 16 kHz or contains "Done" meaning the end of utterance.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D torch.float32 tensor containing the audio samples or
          return None.
        """
        message = await socket.recv()
        if message == "Done":
            return None

        try:
            data = json.loads(message)
            logging.info("Received JSON object: %s", data)
            self.tokenizer, self.options, self.stream_options = self.model.init_options(**data)
            return "settings"
            # Process the JSON object here
        except:
            # Assuming it's an audio byte buffer
            array = np.frombuffer(message, dtype=np.float32)
            return array


def main():
    args = get_args()

    logging.info(vars(args))

    port = args.port
    whisper_model_filename = args.whisper_model_filename
    compute_type = args.compute_type
    device = args.device
    cpu_thread = args.cpu_thread
    min_new_audio_dur=args.min_new_audio_dur

    nn_pool_size = args.nn_pool_size
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    certificate = args.certificate

    if certificate and not Path(certificate).is_file():
        raise ValueError(f"{certificate} does not exist")

    server = StreamingServer(
        whisper_model_filename=whisper_model_filename,
        compute_type=compute_type,
        device=device,
        cpu_thread=cpu_thread,
        min_new_audio_dur=min_new_audio_dur,
        nn_pool_size=nn_pool_size,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        certificate=certificate,
    )
    asyncio.run(server.run(port))

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.DEBUG)
    main()
