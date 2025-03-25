from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue

# Load the SNAC model and move it to GPU.
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda"
model = model.to(snac_device)

def convert_to_audio(multiframe, count):
    """
    Convert a sequence of tokens (multiframe) into an audio byte stream.
    This version uses Python lists for efficiency and is intended to run in parallel.
    """
    if len(multiframe) < 7:
        return None
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    codes_0 = []
    codes_1 = []
    codes_2 = []
    for j in range(num_frames):
        i = 7 * j
        codes_0.append(frame[i])
        codes_1.append(frame[i+1])
        codes_1.append(frame[i+4])
        codes_2.append(frame[i+2])
        codes_2.append(frame[i+3])
        codes_2.append(frame[i+5])
        codes_2.append(frame[i+6])
    codes_0_tensor = torch.tensor(codes_0, device=snac_device, dtype=torch.int32).unsqueeze(0)
    codes_1_tensor = torch.tensor(codes_1, device=snac_device, dtype=torch.int32).unsqueeze(0)
    codes_2_tensor = torch.tensor(codes_2, device=snac_device, dtype=torch.int32).unsqueeze(0)
    codes = [codes_0_tensor, codes_1_tensor, codes_2_tensor]
    # Validate token ranges.
    for code in codes:
        if torch.any(code < 0) or torch.any(code > 4096):
            return None
    with torch.inference_mode():
        audio_hat = model.decode(codes)
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    """
    Processes a token string to extract a numeric ID.
    """
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1:
        print("No token found in the string")
        return None
    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None
  
async def tokens_decoder(token_gen):
    """
    Asynchronously process tokens from the generator.
    Every time 7 new tokens are available (beyond an initial offset), offload the
    conversion to audio to a separate thread.
    """
    buffer = []
    count = 0
    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is None:
            continue
        if token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                # Offload conversion to a background thread so it runs in parallel.
                audio_samples = await asyncio.to_thread(convert_to_audio, buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen):
    """
    Synchronous wrapper for the asynchronous tokens decoder.
    """
    audio_queue = queue.Queue()

    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel value.

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()
