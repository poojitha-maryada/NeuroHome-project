NEUROHOME BACKEND

Model:
D:\dataset\models\p300_best_eegnet.keras

Install:
py -3.13 -m pip install -r requirements.txt

Start:
py -3.13 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

Exact event-aware inference:
POST /predict with BOTH the matching .bdf and *_events.tsv.

The backend mirrors training: 0.1-30 Hz bandpass, 50 Hz notch, average
reference, event sample = round(onset*2048), resample 256 Hz, -200..800 ms
epochs, baseline -200..0 ms, Target=2, NonTarget=1, 100..700 ms model
window, per-epoch normalization, threshold 0.40.
