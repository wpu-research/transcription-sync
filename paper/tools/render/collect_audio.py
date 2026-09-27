"""
Section V-C test material collection: batch-generates spoken responses from
the SAME Gemini Live native-audio model, voice, and persona as the deployed
A.T.L.A.S system (gemini-2.5-flash-native-audio-latest, voice "Charon"),
across five task types. Saves per utterance:

    <outdir>/<id>.wav    24 kHz mono 16-bit PCM (the model's native output)
    <outdir>/<id>.txt    the model's own output transcription (the reference
                         transcript for the MFA oracle and the RATE condition,
                         exactly as the live system would receive it)
    <outdir>/manifest.json

Text prompts (rather than live microphone turns) are used purely as the
elicitation mechanism; the audio is the same native-audio stream the live
system plays. Frame this honestly in the paper (Section V-B).

Usage:
    python collect_audio.py --outdir paper/eval_audio [--per-type 10]
"""
import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from google import genai
from google.genai import types

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-latest"
OUT_SR = 24000

# Five task types mirroring deployed A.T.L.A.S usage (Section III).
PROMPTS = {
    'qa': [  # factual question answering
        "What causes the seasons on Earth? Answer in two or three sentences.",
        "How does a refrigerator keep food cold? Keep it brief.",
        "Why is the sky blue during the day?",
        "What is the difference between RAM and storage in a computer?",
        "How do vaccines work, in simple terms?",
        "Why do we see phases of the moon?",
        "What makes an airplane stay in the air?",
        "How does GPS know where I am?",
        "Why does bread rise when baking?",
        "What is machine learning in one short paragraph?",
    ],
    'task': [  # tool / desktop task narration
        "Imagine you just opened my calendar app. Tell me what you would do to schedule a meeting for tomorrow at 3 pm.",
        "Walk me through how you would rename twenty files on my desktop.",
        "Describe the steps you would take to back up my documents folder.",
        "Tell me how you would set a reminder for my doctor's appointment next Friday.",
        "Explain what you are doing as you organize my downloads folder by file type.",
        "Describe how you would draft and send an email to my team about Monday's deadline.",
        "Tell me the steps to free up disk space on my laptop.",
        "Narrate how you would install a new application safely on my computer.",
        "Describe how you would search my notes for everything about the Berlin trip.",
        "Tell me how you would mute my notifications for the next two hours.",
    ],
    'chat': [  # conversational small talk
        "Good morning! How are you doing today?",
        "I'm feeling a bit tired this afternoon. Any suggestions?",
        "Tell me something interesting I probably don't know.",
        "What's a good way to spend a rainy Sunday?",
        "I just finished a big project at work. Say something encouraging.",
        "Do you ever get bored waiting for me to ask questions?",
        "Recommend a movie for tonight, something light.",
        "I can't decide between tea and coffee right now. Help me choose.",
        "Tell me a short fun fact about space.",
        "What would you do with a free weekend if you were human?",
    ],
    'howto': [  # instructions / how-to
        "Give me quick instructions for making a good cup of pour-over coffee.",
        "How do I jump-start a car safely? Short version.",
        "Teach me the basic steps of making fresh pasta at home.",
        "How should I water and care for a basil plant on my windowsill?",
        "Give me a beginner's three-step plan to start running regularly.",
        "How do I get red wine out of a white shirt?",
        "Explain how to tie a bowline knot in simple steps.",
        "How do I take better photos with my phone? Three tips.",
        "Walk me through stretching exercises for lower back pain.",
        "How do I make my home Wi-Fi faster? A few quick tips.",
    ],
    'summary': [  # summarization / reading back
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "Give me a one-paragraph summary of how the internet works.",
        "Summarize the main idea of the theory of evolution briefly.",
        "In a few sentences, what was the industrial revolution?",
        "Briefly summarize what photosynthesis does for a plant.",
        "Give me the gist of the story of the Trojan horse.",
        "Summarize in plain words what inflation means for my savings.",
        "In three sentences, what is the water cycle?",
        "Briefly explain what the Renaissance changed in Europe.",
        "Summarize how the human immune system fights a cold.",
    ],
}


def _get_api_key() -> str:
    return json.loads((REPO_ROOT / 'config' / 'api_keys.json').read_text())['gemini_api_key']


def live_config():
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=(
            "You are A.T.L.A.S, a helpful spoken desktop assistant. "
            "Answer naturally and conversationally in English, in roughly "
            "two to five spoken sentences. Do not use markdown or lists."
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
    )


async def collect_one(client, prompt: str, wav_path: Path, txt_path: Path,
                       max_seconds: float = 30.0) -> dict | None:
    pcm = bytearray()
    transcript = []
    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=live_config()) as session:
            await session.send_client_content(
                turns=types.Content(role='user', parts=[types.Part(text=prompt)]))
            async for resp in session.receive():
                sc = resp.server_content
                if sc is None:
                    continue
                if sc.output_transcription and sc.output_transcription.text:
                    transcript.append(sc.output_transcription.text)
                if sc.model_turn:
                    for part in (sc.model_turn.parts or []):
                        if part.inline_data and part.inline_data.data:
                            pcm.extend(part.inline_data.data)
                if sc.turn_complete:
                    break
                if len(pcm) > max_seconds * OUT_SR * 2:
                    break
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {str(e)[:120]}')
        return None

    if len(pcm) < OUT_SR:  # < 1 s of audio -> reject
        print(f'  too short ({len(pcm)/2/OUT_SR:.2f}s), skipping')
        return None

    with wave.open(str(wav_path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_SR)
        w.writeframes(bytes(pcm))
    text = ''.join(transcript).strip()
    txt_path.write_text(text)
    return {'wav': wav_path.name, 'transcript': text,
            'duration_s': round(len(pcm) / 2 / OUT_SR, 2)}


async def main_async(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=_get_api_key(), http_options={'api_version': 'v1beta'})

    manifest_path = outdir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for task_type, prompts in PROMPTS.items():
        for i, prompt in enumerate(prompts[:args.per_type]):
            uid = f'{task_type}{i:02d}'
            if uid in manifest:
                continue  # resume support
            wav_path = outdir / f'{uid}.wav'
            print(f'[{uid}] {prompt[:60]}...')
            entry = await collect_one(client, prompt, wav_path, outdir / f'{uid}.txt')
            if entry:
                entry['task_type'] = task_type
                entry['prompt'] = prompt
                manifest[uid] = entry
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"  ok: {entry['duration_s']}s  \"{entry['transcript'][:60]}...\"")
            await asyncio.sleep(args.delay)

    n = len(manifest)
    total = sum(e['duration_s'] for e in manifest.values())
    print(f'\nDone: {n} utterances, {total:.0f}s total audio -> {outdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--per-type', type=int, default=10)
    ap.add_argument('--delay', type=float, default=2.0,
                    help='seconds between requests (rate-limit kindness)')
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
