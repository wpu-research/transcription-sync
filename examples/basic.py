"""Minimal example: drive ARKit mouth blendshapes from a text stream.

Run:  python examples/basic.py
"""
from transcription_sync import VisemeStream


def main():
    vs = VisemeStream(fps=60)

    # Simulate a transcription stream arriving in chunks from an LLM.
    chunks = ["Hello ", "there, ", "how are ", "you today?"]
    vs.set_amplitude(0.3)  # in a real app, update this from the audio envelope

    print("frame_idx  JawOpen  MouthClose  MouthPucker  (14 ARKit channels total)")
    idx = 0
    for chunk in chunks:
        vs.feed_text(chunk)
        for frame in vs.frames():
            print(f"  {idx:4d}     {frame['JawOpen']:.3f}    "
                  f"{frame['MouthClose']:.3f}      {frame['MouthPucker']:.3f}")
            idx += 1

    print(f"\nRendered {idx} viseme frames. "
          f"Feed each frame to your VRM/MetaHuman/ARKit rig at ~13 chars/s.")


if __name__ == "__main__":
    main()
