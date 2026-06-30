"""Voice cloning module.

Builds professional Resemble voice clones from long-form recordings plus the
script they were read from. The hard part — turning multi-minute recordings
into the short, transcribed, per-sentence clips Resemble's dataset format
requires — is done with forced alignment (see ``aligner.py``), so the user
never has to transcribe or hand-cut anything.

Heavy ML dependencies (torch/torchaudio/uroman) live in the optional
``alignment`` Poetry group and are imported lazily inside ``aligner`` so the
API process and the test suite stay lightweight.
"""
