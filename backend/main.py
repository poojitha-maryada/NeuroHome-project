from pathlib import Path
import os, io, tempfile, re
import numpy as np
import pandas as pd
import mne
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

MODEL_PATH = Path(__file__).resolve().parent / "models" / "p300_best_eegnet_fixed.keras"
THRESHOLD = 0.40
DECISION_WINDOW_SEC = 1.5
# The model Target probability is the single confidence/percentage score used by the novelty.
# Candidates first pass THRESHOLD; the highest confidence percentage wins.
CONFIDENCE_DEFINITION = "model Target probability expressed as confidence percentage"

SFREQ = 256
RAW_SFREQ = 2048
TMIN, TMAX = -0.2, 0.8

CHANNELS = ["Fp1","AF3","F7","F3","FC1","FC5","T7","C3","CP1","CP5","P7","P3",
            "Pz","PO3","O1","Oz","O2","PO4","P4","P8","CP6","CP2","C4","T8",
            "FC6","FC2","F4","F8","AF4","Fp2","Fz","Cz"]

# NOVELTY: each 1.5-second window selects its highest-confidence accepted Target.
SUPPORTED_COMMANDS = ["LIGHT ON", "TELEPHONE ON", "WINDOW OPEN", "DOOR OPEN", "TV ON", "RADIO ON"]
COMMAND_TARGET_NAMES = ["Light", "Telephone", "Window", "Door", "TV", "Radio"]
# Map uploaded run numbers to the six existing NeuroHome devices.
# This prevents an undefined RUN_COMMANDS reference when an events file only
# contains Target/NonTarget labels and the appliance identity comes from run.
RUN_COMMANDS = {run: SUPPORTED_COMMANDS[run % len(SUPPORTED_COMMANDS)] for run in range(19)}

def extract_run_number(*filenames):
    for filename in filenames:
        m = re.search(r"(?:^|[_\-])run[-_]?([0-9]+)(?:[_\-.]|$)", str(filename), re.I)
        if m:
            return int(m.group(1))
    return None

app = FastAPI(title="NeuroHome P300 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
model = None

@app.on_event("startup")
def startup():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)

@app.get("/")
def root():
    return {"name":"NeuroHome P300 API","status":"online"}

@app.get("/health")
def health():
    return {"status":"healthy","model":MODEL_PATH.name,"threshold":THRESHOLD,
            "percentage_definition":"model Target probability expressed as percentage",
            "confidence_definition":CONFIDENCE_DEFINITION,
            "channels":32,"training_window":"100-700 ms","sampling_rate":SFREQ,"decision_window_sec":DECISION_WINDOW_SEC}

def model_input(epoch):
    start, end = 26, 180
    x = epoch[:, start:end]
    if x.shape != (32, 154):
        raise ValueError(f"Model input window has shape {x.shape}; expected (32, 154)")
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.astype(np.float32)[None, :, :, None]

def normalize_command(value):
    """Extract a supported smart-home command from an event label."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    if not s:
        return None

    # Exact forms first.
    aliases = {
        "light on":"LIGHT ON", "lights on":"LIGHT ON", "lamp on":"LIGHT ON",
        "light off":"LIGHT OFF", "lights off":"LIGHT OFF", "lamp off":"LIGHT OFF",
        "tv on":"TV ON", "tv off":"TV OFF", "television on":"TV ON", "television off":"TV OFF",
        "telephone on":"TELEPHONE ON", "telephone off":"TELEPHONE OFF", "phone on":"TELEPHONE ON", "phone off":"TELEPHONE OFF",
        "door open":"DOOR OPEN", "open door":"DOOR OPEN",
        "door close":"DOOR CLOSE", "close door":"DOOR CLOSE",
        "radio on":"RADIO ON", "radio off":"RADIO OFF",
        "window open":"WINDOW OPEN", "window close":"WINDOW CLOSE",
    }
    if s in aliases:
        return aliases[s]

    # Also handle labels such as LIGHT_ON, target_light_on, Light-On-Target,
    # or other event labels that contain the appliance/action pair.
    words = set(s.split())
    if "light" in words or "lamp" in words or "lights" in words:
        if "on" in words: return "LIGHT ON"
        if "off" in words: return "LIGHT OFF"
    if "tv" in words or "television" in words:
        if "on" in words: return "TV ON"
        if "off" in words: return "TV OFF"
    if "door" in words:
        if "open" in words: return "DOOR OPEN"
        if "close" in words or "closed" in words: return "DOOR CLOSE"
    if "window" in words:
        if "open" in words: return "WINDOW OPEN"
        if "close" in words or "closed" in words: return "WINDOW CLOSE"
    return None


def read_events(path, run_number=None):
    """Read BIDS-style TSV/CSV event files robustly.

    Supports UTF-8 BOMs, tabs, commas, semicolons and common BIDS columns.
    An explicit appliance command is kept as metadata and is executed only
    when the EEG model accepts that same event as a Target.
    """
    # BIDS events.tsv is tab-separated. Try that explicitly first; then fall
    # back to automatic delimiter detection for CSV/TXT files.
    attempts = [
        dict(sep="\t", encoding="utf-8-sig"),
        dict(sep=None, engine="python", encoding="utf-8-sig"),
        dict(sep=",", encoding="utf-8-sig"),
    ]
    df = None
    last_error = None
    for kwargs in attempts:
        try:
            candidate = pd.read_csv(path, **kwargs)
            candidate.columns = [str(c).replace("\ufeff", "").strip() for c in candidate.columns]
            if len(candidate.columns) > 1:
                df = candidate
                break
            last_error = ValueError("Only one column was detected")
        except Exception as ex:
            last_error = ex
    if df is None:
        raise ValueError(f"Could not read events file: {last_error}")

    low = {str(c).strip().lower(): c for c in df.columns}
    onset_col = next((low[x] for x in ("onset", "onsets", "time", "timestamp") if x in low), None)
    sample_col = next((low[x] for x in ("sample", "samples", "sample_index") if x in low), None)
    value_col = next((low[x] for x in (
        "value", "event", "code", "trial_type", "event_id", "trigger",
        "type", "event_type", "stimulus"
    ) if x in low), None)
    command_col = next((low[x] for x in (
        "command", "appliance_command", "home_command", "appliance",
        "action", "event_label", "command_label"
    ) if x in low), None)

    # Some native event files contain only Target/NonTarget labels. In that
    # case use the BDF annotations as the timing source.
    if onset_col is None and sample_col is None:
        return None
    if value_col is None and command_col is None:
        raise ValueError(
            "events file needs a value/event/code/trial_type column or an explicit command column."
        )

    out = []
    for _, row in df.iterrows():
        # Prefer BIDS onset (seconds); otherwise use the raw sample index.
        try:
            if onset_col is not None:
                onset_seconds = float(row[onset_col])
                raw_sample = int(round(onset_seconds * RAW_SFREQ))
            else:
                raw_sample = int(round(float(row[sample_col])))
        except Exception:
            continue

        raw_value = row[value_col] if value_col is not None else None
        # Prefer an explicit command column, but also scan the whole event row
        # so command labels placed in trial_type/value/event/etc. are recognized.
        raw_command = row[command_col] if command_col is not None else raw_value
        command = normalize_command(raw_command)
        if command is None:
            for cell in row.tolist():
                command = normalize_command(cell)
                if command is not None:
                    break
        # Native EPFLP300 events only contain Target/NonTarget. The appliance
        # identity therefore comes from the uploaded file's run number, not
        # from the TSV label. Explicit command text still takes precedence.
        if command is None and run_number in RUN_COMMANDS:
            command = RUN_COMMANDS[run_number]

        # Determine Target/NonTarget class for the EEG model.
        value = None
        if raw_value is not None:
            try:
                value = int(float(raw_value))
            except Exception:
                text_value = str(raw_value).strip().lower()
                if text_value in ("target", "t", "1", "target event"):
                    value = 2
                elif text_value in ("nontarget", "non-target", "non_target", "nt", "non target", "0"):
                    value = 1

        # Some command-only event files have no separate Target/NonTarget
        # column. In that format the explicit command event itself is the
        # command-bearing Target epoch.
        if value not in (1, 2) and command is not None:
            value = 2

        if value in (1, 2):
            out.append([raw_sample, 0, value, command])

    if not out:
        raise ValueError("No usable EEG events were found in the events file.")
    return out

def prepare(bdf, events_file):
    raw=mne.io.read_raw_bdf(bdf, preload=True, verbose=False)
    if abs(raw.info["sfreq"]-RAW_SFREQ)>1e-6:
        raise ValueError(f"Expected 2048 Hz raw EEG, got {raw.info['sfreq']} Hz.")
    missing=[c for c in CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError("Missing channels: "+", ".join(missing))
    run_number=extract_run_number(bdf.name, events_file.name)
    event_rows=read_events(events_file, run_number)
    if event_rows is None:
        event_rows=[]
        for ann_onset, ann_desc in zip(raw.annotations.onset, raw.annotations.description):
            text=str(ann_desc).strip().lower()
            value = 2 if text in ("2", "target", "t", "target event") else 1 if text in ("1", "0", "nontarget", "non-target", "non_target", "nt", "non target") else None
            if value is not None:
                event_rows.append([int(round(float(ann_onset)*RAW_SFREQ)),0,value,RUN_COMMANDS.get(run_number)])
        if not event_rows:
            raise ValueError("No usable timing information found in the events file or BDF annotations.")
    events=np.asarray([[r[0],r[1],r[2]] for r in event_rows],dtype=int)
    raw.filter(.1,30.,verbose=False)
    raw.notch_filter(50.,verbose=False)
    raw.pick(CHANNELS)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.resample(SFREQ,verbose=False)
    events[:,0]=np.round(events[:,0]*SFREQ/RAW_SFREQ).astype(int)
    epochs=mne.Epochs(raw,events,event_id={"NonTarget":1,"Target":2},
                      tmin=TMIN,tmax=TMAX,baseline=(TMIN,0),
                      preload=True,reject_by_annotation=True,verbose=False)
    X=epochs.get_data(copy=True)
    y=(epochs.events[:,2]==2).astype(int)
    if len(X)==0:
        raise ValueError("No usable epochs remained.")
    # Match command metadata to the exact original event selected by MNE.
    # IMPORTANT: event samples are resampled from 2048 Hz to 256 Hz, so matching
    # a 2048-Hz sample number against epochs.events would otherwise lose every
    # command and produce NO COMMAND for valid command-labelled events.
    all_event_commands=[r[3] for r in event_rows]
    epoch_commands=[all_event_commands[int(idx)] for idx in epochs.selection]
    epoch_onsets = epochs.events[:,0].astype(float) / SFREQ
    return np.stack([model_input(e)[0] for e in X]), y, epoch_commands, run_number, epoch_onsets

def apply_command(state, command):
    """Apply one supported command to its mapped smart-home device."""
    mapping = {
        "LIGHT ON": "light",
        "LIGHT OFF": "light",
        "TELEPHONE ON": "telephone",
        "TELEPHONE OFF": "telephone",
        "WINDOW OPEN": "window",
        "WINDOW CLOSE": "window",
        "TV ON": "tv",
        "TV OFF": "tv",
        "DOOR OPEN": "door",
        "DOOR CLOSE": "door",
        "RADIO ON": "radio",
        "RADIO OFF": "radio",
    }
    device = mapping.get(command)
    if device is None:
        return None, None
    state[device] = True
    return device, True

@app.post("/predict")
async def predict(eeg_file: UploadFile=File(...), events_file: UploadFile=File(...)):
    if not eeg_file.filename.lower().endswith(".bdf"):
        raise HTTPException(400,"EEG file must be .bdf")
    if not events_file.filename.lower().endswith((".tsv",".txt",".csv")):
        raise HTTPException(400,"Events file must be .tsv/.txt/.csv")
    with tempfile.TemporaryDirectory() as td:
        b=Path(td)/Path(eeg_file.filename).name
        e=Path(td)/Path(events_file.filename).name
        b.write_bytes(await eeg_file.read()); e.write_bytes(await events_file.read())
        try:
            X,y,event_commands,run_number,epoch_onsets=prepare(b,e)
            p=model.predict(X,verbose=0).reshape(-1)
            pred=(p>=THRESHOLD).astype(int)
            results=[]
            appliance_state={"light":False,"telephone":False,"window":False,"door":False,"tv":False,"radio":False}
            appliance_actions=[]

            # NOVELTY: divide the prediction timeline into non-overlapping 1.5-second
            # decision windows. Collect every Target signal at/above the threshold,
            # calculate its percentage, choose the highest-percentage signal, and
            # execute that winner once. The selected event command maps only to one of the six existing devices.
            if len(epoch_onsets):
                window_starts = np.floor((epoch_onsets - epoch_onsets[0]) / DECISION_WINDOW_SEC) * DECISION_WINDOW_SEC + epoch_onsets[0]
            else:
                window_starts = np.array([])
            selected_epochs=set()
            decision_windows=[]
            for ws in sorted(set(window_starts.tolist())):
                idxs=[i for i,w in enumerate(window_starts) if abs(w-ws)<1e-9]
                candidates=[i for i in idxs if pred[i]==1 and p[i]>=THRESHOLD]
                decision_windows.append({"start_sec":round(float(ws),3),"end_sec":round(float(ws+DECISION_WINDOW_SEC),3),"candidates":[{"epoch":i+1,"confidence":round(float(p[i])*100,1)} for i in candidates],"selected_epoch":None,"selected_confidence":None,"executed_command":None})
                if candidates:
                    best=max(candidates,key=lambda i: float(p[i]))
                    confidence_pct=round(float(p[best])*100,1)
                    decision_windows[-1]["selected_epoch"]=best+1
                    decision_windows[-1]["selected_confidence"]=confidence_pct
                    selected_epochs.add(best)
                    effective_command=event_commands[best] if event_commands[best] in SUPPORTED_COMMANDS else None
                    device,state=apply_command(appliance_state,effective_command) if effective_command else (None,None)
                    if device:
                        appliance_actions.append({
                            "epoch":best+1,"device":device,"state":bool(state),
                            "command":effective_command,"confidence":confidence_pct,
                            "window_start_sec":round(float(ws),3),
                            "window_end_sec":round(float(ws+DECISION_WINDOW_SEC),3),
                        })
                        decision_windows[-1]["executed_command"]=effective_command

            for i,(q,z,yy,cmd) in enumerate(zip(p,pred,y,event_commands)):
                results.append({
                    "epoch":i+1,
                    "probability":float(q),
                    "prediction":"Target" if z else "NonTarget",
                    "true_event":"Target" if yy else "NonTarget",
                    "correct":bool(z==yy),
                    "event_command":cmd,
                    "command": appliance_actions[[a["epoch"] for a in appliance_actions].index(i+1)]["command"] if i+1 in [a["epoch"] for a in appliance_actions] else "NO COMMAND",
                    "executed":i in selected_epochs,
                    "accepted":bool(z and q>=THRESHOLD),
                    "confidence":round(float(q)*100,1),
                    "decision_window_start_sec":round(float(window_starts[i]),3) if len(window_starts) else None,
                    "selected_for_execution":i in selected_epochs
                })

            last_action=appliance_actions[-1] if appliance_actions else None
            has_explicit_commands=any(x is not None for x in event_commands)
            return {
                "eeg_file":eeg_file.filename,"events_file":events_file.filename,
                "epochs":len(results),"predicted_target_count":int(pred.sum()),
                "predicted_nontarget_count":int((pred==0).sum()),
                "true_target_count":int(y.sum()),"threshold":THRESHOLD,
                "decision_window_sec":DECISION_WINDOW_SEC,
                "selection_rule":"collect all Target signals >= threshold in each 1.5-second window, use model confidence percentage as the single score, select the highest score, and execute once",
                "confidence_definition":CONFIDENCE_DEFINITION,
                "supported_commands":SUPPORTED_COMMANDS,
                "appliance_state":appliance_state,
                "appliance_actions":appliance_actions,
                "decision_windows":decision_windows,
                "executions":len(appliance_actions),
                "last_bci_command":last_action["command"] if last_action else "NO COMMAND",
                "last_bci_confidence":last_action["confidence"] if last_action else None,
                "completed":False,
                "predictions":results
            }
        except Exception as ex:
            raise HTTPException(400,str(ex))

@app.post("/predict-epoch")
async def predict_epoch(file: UploadFile=File(...)):
    if not file.filename.lower().endswith(".npy"):
        raise HTTPException(400,"Upload a .npy epoch file.")
    arr=np.load(io.BytesIO(await file.read()))
    if arr.shape!=(32,256): raise HTTPException(400,f"Expected (32,256), got {arr.shape}")
    p=float(model.predict(model_input(arr),verbose=0).reshape(-1)[0])
    accepted = p >= THRESHOLD
    # This endpoint has no appliance identity, so it reports only the EEG result.
    return {"probability":p,"confidence":round(p*100,1),
            "confidence_label":"HIGH" if p>=0.80 else ("MEDIUM" if p>=0.60 else "LOW"),
            "label":"Target" if accepted else "NonTarget",
            "command":"NO COMMAND","threshold":THRESHOLD}
