"""Phoneme/grapheme -> ARKit viseme table (articulatory-class priors).

The values are mouth-region ARKit blendshape activations in [0, 1], chosen from
articulatory phonetics: bilabials close the lips (MouthClose), rounded vowels
funnel/pucker, spread vowels stretch, labiodentals raise the upper lip, etc.
This is the deterministic, training-free viseme layer of TranscriptionSync.

These are *design constants*, not fitted to any corpus.
"""

# The 14 mouth-region ARKit channels this engine drives. All other ARKit
# channels (eyes, brows, head) are left untouched for an additive face rig.
CHANNELS = (
    "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker",
    "MouthStretchLeft", "MouthStretchRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthShrugUpper", "MouthRollLower",
    "MouthDimpleLeft", "MouthDimpleRight",
)

# Grapheme -> target blendshape pose. Lowercase letters; "_" is the default
# consonant fallback. Mapping graphemes (not phonemes) keeps the layer G2P-free
# and language-agnostic for shallow-orthography input; pair with a real G2P
# front-end for deeper orthographies.
VISEME: dict[str, dict[str, float]] = {
    # Vowels
    "a": {"JawOpen": 0.60, "MouthLowerDownLeft": 0.35, "MouthLowerDownRight": 0.35, "MouthUpperUpLeft": 0.18, "MouthUpperUpRight": 0.18},
    "e": {"JawOpen": 0.32, "MouthStretchLeft": 0.42, "MouthStretchRight": 0.42, "MouthLowerDownLeft": 0.18, "MouthLowerDownRight": 0.18},
    "i": {"JawOpen": 0.18, "MouthStretchLeft": 0.52, "MouthStretchRight": 0.52},
    "o": {"JawOpen": 0.42, "MouthFunnel": 0.38, "MouthPucker": 0.18, "MouthLowerDownLeft": 0.22, "MouthLowerDownRight": 0.22},
    "u": {"JawOpen": 0.18, "MouthPucker": 0.58, "MouthFunnel": 0.42},
    # Bilabials -- full lip closure
    "m": {"JawOpen": 0.00, "MouthClose": 0.92, "MouthShrugUpper": 0.15},
    "b": {"JawOpen": 0.04, "MouthClose": 0.65},
    "p": {"JawOpen": 0.02, "MouthClose": 0.80},
    # Labiodental
    "f": {"JawOpen": 0.06, "MouthUpperUpLeft": 0.58, "MouthUpperUpRight": 0.58, "MouthLowerDownLeft": 0.12, "MouthLowerDownRight": 0.12},
    "v": {"JawOpen": 0.08, "MouthUpperUpLeft": 0.48, "MouthUpperUpRight": 0.48},
    # Sibilants
    "s": {"JawOpen": 0.07, "MouthStretchLeft": 0.28, "MouthStretchRight": 0.28},
    "z": {"JawOpen": 0.10, "MouthStretchLeft": 0.22, "MouthStretchRight": 0.22},
    # Rounded
    "w": {"JawOpen": 0.14, "MouthPucker": 0.50, "MouthFunnel": 0.38},
    "r": {"JawOpen": 0.20, "MouthFunnel": 0.18, "MouthLowerDownLeft": 0.14, "MouthLowerDownRight": 0.14},
    # Alveolar
    "l": {"JawOpen": 0.16, "MouthLowerDownLeft": 0.10, "MouthLowerDownRight": 0.10},
    "t": {"JawOpen": 0.10, "MouthStretchLeft": 0.12, "MouthStretchRight": 0.12},
    "d": {"JawOpen": 0.13, "MouthStretchLeft": 0.10, "MouthStretchRight": 0.10},
    "n": {"JawOpen": 0.10},
    # Fricatives / others
    "h": {"JawOpen": 0.28},
    "j": {"JawOpen": 0.14, "MouthStretchLeft": 0.18, "MouthStretchRight": 0.18},
    "k": {"JawOpen": 0.20}, "g": {"JawOpen": 0.22}, "x": {"JawOpen": 0.15},
    "c": {"JawOpen": 0.10, "MouthStretchLeft": 0.15, "MouthStretchRight": 0.15},
    "q": {"JawOpen": 0.18}, "y": {"JawOpen": 0.14, "MouthStretchLeft": 0.20, "MouthStretchRight": 0.20},
    # Default consonant
    "_": {"JawOpen": 0.12, "MouthShrugUpper": 0.08},
}

# Brief labial closure rendered between words / on punctuation.
WORD_PAUSE = {"JawOpen": 0.04, "MouthClose": 0.28}

# Characters treated as inter-word pauses (rendered as WORD_PAUSE).
PAUSE_CHARS = frozenset(" ,.!?;:")
