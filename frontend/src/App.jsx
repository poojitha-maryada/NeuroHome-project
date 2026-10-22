import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const API = (import.meta.env.VITE_API_URL || "https://neurohome-backend-so4i.onrender.com").replace(/\/$/, "");

const MODEL = {
  accuracy: "70.07%",
  balanced: "69.25%",
  targetF1: "43.59%",
  threshold: "0.40",
  channels: 32,
  runs: 6,
  epochs: 838,
  window: "100-700 ms"
};

const DEVICES = [
  { id: "tv", name: "Television" },
  { id: "telephone", name: "Telephone" },
  { id: "light", name: "Lamp" },
  { id: "door", name: "Door" },
  { id: "window", name: "Window" },
  { id: "radio", name: "Radio" }
];

function safePct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(1) : "-";
}

export default function App() {
  const [page, setPage] = useState("Dashboard");
  const [eegFile, setEegFile] = useState(null);
  const [eventsFile, setEventsFile] = useState(null);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [devices, setDevices] = useState({
    tv: false,
    telephone: false,
    light: false,
    door: false,
    window: false,
    radio: false
  });

  async function predict() {
    if (!eegFile || !eventsFile) {
      setError("Please select both the EEG BDF and EVENTS TSV files.");
      return;
    }

    setBusy(true);
    setError("");

    const form = new FormData();
    form.append("eeg_file", eegFile);
    form.append("events_file", eventsFile);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        body: form
      });

      const text = await res.text();

      let json;
      try {
        json = JSON.parse(text);
      } catch {
        throw new Error(
          `Backend returned an invalid response (${res.status}).`
        );
      }

      if (!res.ok) {
        throw new Error(json.detail || "Prediction failed.");
      }

      const synced = {
        tv: false,
        telephone: false,
        light: false,
        door: false,
        window: false,
        radio: false,
        ...(json.appliance_state || {})
      };

      setData(json);
      setDevices(synced);
      setPage("Smart Home");

      // Voice feedback for the last executed appliance command.
      if ("speechSynthesis" in window && Array.isArray(json.appliance_actions) && json.appliance_actions.length > 0) {
        const action = json.appliance_actions[json.appliance_actions.length - 1];
        const deviceNames = {
          light: "Lamp",
          tv: "Television",
          telephone: "Telephone",
          door: "Door",
          window: "Window",
          radio: "Radio"
        };
        const deviceName = deviceNames[action.device] || action.device || "Appliance";
        const confidence = Number(action.confidence);
        const stateText = action.state ? "activated" : "deactivated";
        const utterance = new SpeechSynthesisUtterance(
          `${deviceName} ${stateText}. Confidence: ${Number.isFinite(confidence) ? confidence + " percent" : "not available"}.`
        );
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }
    } catch (e) {
      setError(
        e?.message || "Unable to connect to the NeuroHome backend."
      );
    } finally {
      setBusy(false);
    }
  }

  const predictions = data?.predictions || [];

  const latest =
    predictions.length > 0
      ? predictions[predictions.length - 1]
      : null;

  const actions = data?.appliance_actions || [];

  const lastAccepted =
    actions.length > 0
      ? actions[actions.length - 1]
      : null;

  const targetPct = data?.epochs
    ? (
        (Number(data.predicted_target_count || 0) /
          Number(data.epochs)) *
        100
      ).toFixed(1)
    : "-";

  const activeCount = Object.values(devices).filter(Boolean).length;

  const content = useMemo(() => {
    if (page === "EEG Signal") {
      return (
        <EEGPage
          eegFile={eegFile}
          eventsFile={eventsFile}
          data={data}
          busy={busy}
          error={error}
          setEegFile={setEegFile}
          setEventsFile={setEventsFile}
          predict={predict}
        />
      );
    }

    if (page === "Brain Response") {
      return <BrainPage data={data} />;
    }

    if (page === "Smart Home") {
      return (
        <SmartHome
          devices={devices}
          activeCount={activeCount}
          lastAccepted={lastAccepted}
          data={data}
        />
      );
    }

    if (page === "Model Results") {
      return <ModelPage />;
    }

    return (
      <Dashboard
        data={data}
        targetPct={targetPct}
        latest={latest}
        setPage={setPage}
      />
    );
  }, [
    page,
    data,
    latest,
    lastAccepted,
    eegFile,
    eventsFile,
    busy,
    error,
    devices,
    activeCount,
    targetPct
  ]);

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="logo">
          NEURO<span>HOME</span>
        </div>

        <div className="sideTag">P300 BCI SYSTEM</div>

        <nav>
          {[
            "Dashboard",
            "EEG Signal",
            "Brain Response",
            "Smart Home",
            "Model Results"
          ].map((item) => (
            <button
              key={item}
              className={`navItem ${page === item ? "active" : ""}`}
              onClick={() => setPage(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="sideBottom">
          <div className="backend">
            <i />
            Backend connected
          </div>

          <div className="modelMini">
            EEGNet - 32 channels
            <br />
            100-700 ms - adaptive threshold 40%-70%
          </div>
        </div>
      </aside>

      <main className="mainArea">
        <header className="topbar">
          <div>
            <div className="pageKicker">
              NEUROHOME / {page.toUpperCase()}
            </div>
            <h2>{page}</h2>
          </div>

          <div className="live">
            <i /> SYSTEM ONLINE
          </div>
        </header>

        <div className="content">{content}</div>
      </main>
    </div>
  );
}

function Dashboard({ data, targetPct, latest, setPage }) {
  return (
    <>
      <section className="heroGrid">
        <div className="heroCopy">
          <div className="eyebrow">BRAIN-COMPUTER INTERFACE</div>

          <h1>NEUROHOME - BRAIN-CONTROLLED SMART HOME</h1>

          <p>P300-based real-time appliance control.</p>

          <button
            className="primary"
            onClick={() => setPage("EEG Signal")}
          >
            UPLOAD REAL EEG <b aria-hidden="true">→</b>
          </button>
        </div>

        <div className="brainCard">
          <div className="brainRing">
            <div>
              EEG
              <br />
              <strong>P300</strong>
            </div>
          </div>

          <div className="brainMeta">
            <span>MODEL</span>
            <strong>EEGNet</strong>

            <span>STATUS</span>
            <strong className="green">READY</strong>
          </div>
        </div>
      </section>

      <section className="statsGrid">
        <Stat
          label="ACCURACY"
          value={MODEL.accuracy}
          sub="Untouched Run 5"
        />

        <Stat
          label="BALANCED ACC."
          value={MODEL.balanced}
          sub="Untouched Run 5"
        />

        <Stat
          label="TARGET F1"
          value={MODEL.targetF1}
          sub="P300 target class"
        />

        <Stat
          label="PREDICTED TARGETS"
          value={data?.predicted_target_count ?? "-"}
          sub={
            data ? `${targetPct}% of epochs` : "Run an EEG analysis"
          }
        />
      </section>

      <section className="twoCol">
        <Panel title="SYSTEM FLOW">
          <div className="flow">
            <Flow n="01" t="REAL EEG" />
            <b aria-hidden="true">→</b>
            <Flow n="02" t="PREPROCESS" />
            <b aria-hidden="true">→</b>
            <Flow n="03" t="EEGNet" />
            <b aria-hidden="true">→</b>
            <Flow n="04" t="TARGET / NON-TARGET" />
            <b aria-hidden="true">→</b>
            <Flow n="05" t="ONE APPLIANCE" />
          </div>
        </Panel>

        <Panel title="LATEST COMMAND">
          <div className="commandBox">
            <span>{latest?.command || "WAITING FOR EEG"}</span>

            <small>
              {latest
                ? `${safePct(Number(latest.probability) * 100)}% model probability`
                : "No prediction yet"}
            </small>
          </div>
        </Panel>
      </section>

      <section className="datasetStrip">
        <div>
          <span>DATASET</span>
          <strong>6 runs</strong>
        </div>

        <div>
          <span>USABLE EPOCHS</span>
          <strong>838</strong>
        </div>

        <div>
          <span>CHANNELS</span>
          <strong>32</strong>
        </div>

        <div>
          <span>MODEL WINDOW</span>
          <strong>100-700 ms</strong>
        </div>

        <div>
          <span>ADAPTIVE THRESHOLD</span>
          <strong>40%-70%</strong>
        </div>
      </section>

      <NoveltyPanel data={data} />
    </>
  );
}

function EEGPage({
  eegFile,
  eventsFile,
  data,
  busy,
  error,
  setEegFile,
  setEventsFile,
  predict
}) {
  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">REAL EEG DATASET</div>

        <h1>Run the trained model.</h1>

        <p>
          Upload the matching BDF and events file. EEG predictions
          are evaluated inside 1.5-second decision windows using
          an adaptive 40%-70% confidence threshold.
        </p>
      </section>

      <div className="uploadPanel">
        <label>
          <span>EEG BDF</span>

          <input
            type="file"
            accept=".bdf"
            onChange={(e) =>
              setEegFile(e.target.files?.[0] || null)
            }
          />

          {eegFile && <em>{eegFile.name}</em>}
        </label>

        <label>
          <span>EVENTS TSV</span>

          <input
            type="file"
            accept=".tsv,.txt,.csv"
            onChange={(e) =>
              setEventsFile(e.target.files?.[0] || null)
            }
          />

          {eventsFile && <em>{eventsFile.name}</em>}
        </label>

        <button
          className="primary analyze"
          onClick={predict}
          disabled={!eegFile || !eventsFile || busy}
        >
          {busy ? "ANALYZING..." : "ANALYZE EEG →"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="infoGrid">
        <Info label="RAW SAMPLING" value="2048 Hz" />
        <Info label="MODEL SAMPLING" value="256 Hz" />
        <Info label="EPOCH" value="-0.2 to 0.8 s" />
        <Info label="MODEL WINDOW" value="100-700 ms" />
      </div>

      {data && <ResultSummary data={data} />}
    </>
  );
}

function BrainPage({ data }) {
  const selected = data?.appliance_actions?.length
    ? data.appliance_actions[data.appliance_actions.length - 1]
    : null;

  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">BRAIN RESPONSE</div>

        <h1>P300 prediction timeline.</h1>

        <p>
          The model classifies each EEG epoch as Target or
          Non-Target.
        </p>
      </section>

      <div className="responseCards">
        <Stat
          label="TARGETS"
          value={data?.predicted_target_count ?? "-"}
          sub="P300 detected"
        />

        <Stat
          label="NON-TARGETS"
          value={data?.predicted_nontarget_count ?? "-"}
          sub="background events"
        />

        <Stat
          label="CONFIDENCE"
          value={
            selected
              ? `${safePct(selected.confidence)}%`
              : "-"
          }
          sub={
            selected
              ? "Final selected command"
              : "Awaiting EEG"
          }
        />
      </div>

      <NoveltyPanel data={data} />
    </>
  );
}

function NoveltyPanel({ data }) {
  const windows =
    data?.decision_windows ||
    data?.windows ||
    [];

  const highestWindowIndex = windows.reduce((bestIndex, w, i) => {
    const confidence = Number(w?.selected_confidence);

    if (
      w?.selected_confidence === null ||
      w?.selected_confidence === undefined ||
      !Number.isFinite(confidence)
    ) {
      return bestIndex;
    }

    if (bestIndex === -1) return i;

    const bestConfidence = Number(windows[bestIndex]?.selected_confidence);
    return confidence > bestConfidence ? i : bestIndex;
  }, -1);

  return (
    <section className="panel noveltyPanel">
      <div className="panelTitle">
        1.5-SECOND DECISION WINDOW
      </div>

      <p style={{ marginBottom: "18px" }}>
        All accepted EEG signals are shown with their confidence.
      </p>

      {windows.length === 0 ? (
        <div className="empty">
          Analyze EEG to display signals from the
          1.5-second window.
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "13px"
            }}
          >
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  1.5 S WINDOW
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  ADAPTIVE THRESHOLD
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  WINDOW MEAN
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  CONFIDENCE
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  EPOCH
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  STATUS
                </th>
              </tr>
            </thead>

            <tbody>
              {windows.map((w, i) => {
                const selectedConfidence =
                  w.selected_confidence !== null &&
                  w.selected_confidence !== undefined
                    ? Number(w.selected_confidence)
                    : null;
                const isGlobalHighest = i === highestWindowIndex;

                return (
                  <tr key={i}>
                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "700"
                      }}
                    >
                      {w.start_sec ?? "-"}s - {w.end_sec ?? "-"}s
                    </td>

                    <td style={{ padding: "12px 10px" }}>
                      {w.adaptive_threshold ?? "-"}%
                    </td>

                    <td style={{ padding: "12px 10px" }}>
                      {w.window_mean_confidence ?? "-"}%
                    </td>

                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "800",
                        color: isGlobalHighest ? "#82e2ae" : "inherit"
                      }}
                    >
                      {selectedConfidence === null || !Number.isFinite(selectedConfidence)
                        ? "-"
                        : (
                          <>
                            {safePct(selectedConfidence)}%
                            {isGlobalHighest && (
                              <span style={{
                                display: "block",
                                fontSize: "10px",
                                letterSpacing: "1px",
                                marginTop: "4px"
                              }}>
                                HIGHEST CONFIDENCE
                              </span>
                            )}
                          </>
                        )}
                    </td>

                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "800"
                      }}
                    >
                      {w.selected_epoch
                        ? `EPOCH ${w.selected_epoch}`
                        : "NONE"}
                    </td>

                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "800"
                      }}
                    >
                      {w.selected_epoch
                        ? "EXECUTED ONCE"
                        : "NO ACCEPTED TARGET"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function SmartHome({
  devices,
  activeCount,
  lastAccepted,
  data
}) {
  const activeDevices = DEVICES.filter(
    (d) => Boolean(devices[d.id])
  );

  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">AUTOMATIC SMART HOME</div>

        <h1>Six appliances. One EEG command at a time.</h1>

        <p>
          The selected EEG Target controls only its mapped
          appliance. One execution is made per 1.5-second
          decision window.
        </p>
      </section>

      <div className="homeHeader">
        <div>
          <span>ACTIVE DEVICES</span>
          <strong>{activeCount} / 6</strong>
        </div>

        <div>
          <span>LAST BCI COMMAND</span>
          <strong>
            {lastAccepted?.command || "WAITING FOR EEG"}
          </strong>
        </div>

        <div>
          <span>CONFIDENCE</span>
          <strong>
            {lastAccepted
              ? `${safePct(lastAccepted.confidence)}%`
              : "-"}
          </strong>
        </div>
      </div>

      <WholeHomeFloorPlan devices={devices} />

      <section className="activeNow">
        <div className="panelTitle">ACTIVE NOW</div>

        {activeDevices.length ? (
          <div className="activeList">
            {activeDevices.map((d) => (
              <div className="activeItem" key={d.id}>
                <b>{d.name}</b>
                <span>
                  {d.id === "door" || d.id === "window"
                    ? "OPEN"
                    : "ON"}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">
            No appliance is active. All six appliances are
            OFF/CLOSED.
          </div>
        )}
      </section>

      <div className="deviceGrid">
        {DEVICES.map((d) => (
          <div
            className={`device device-${d.id} ${
              devices[d.id] ? "on" : ""
            }`}
            key={d.id}
          >
            <DeviceVisual
              type={d.id}
              on={devices[d.id]}
            />

            <div>
              <h3>{d.name}</h3>

              <small>
                {d.id === "door" || d.id === "window"
                  ? devices[d.id]
                    ? "OPEN"
                    : "CLOSED"
                  : devices[d.id]
                    ? "ON"
                    : "OFF"}
              </small>
            </div>

            <div
              className={`autoState ${
                devices[d.id] ? "active" : ""
              }`}
            >
              {devices[d.id] ? "ACTIVE" : "STANDBY"}
            </div>
          </div>
        ))}
      </div>

      <NoveltyPanel data={data} />

      <div className="mapping">
        <span>RUN 1 <b>TELEVISION</b></span>
        <span>RUN 2 <b>TELEPHONE</b></span>
        <span>RUN 3 <b>LIGHT</b></span>
        <span>RUN 4 <b>DOOR</b></span>
        <span>RUN 5 <b>WINDOW</b></span>
        <span>RUN 6 <b>RADIO</b></span>
      </div>
    </>
  );
}

function WholeHomeFloorPlan({ devices }) {
  const mountRef = useRef(null);
  const state = (id) => Boolean(devices && devices[id]);

  useEffect(() => {
    const host = mountRef.current;
    if (!host) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#d9dedc");
    scene.fog = new THREE.Fog("#d9dedc", 26, 48);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    camera.position.set(18, 23, 25);
    camera.lookAt(0, 0.55, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.domElement.className = "neurohome-3d-canvas";
    renderer.domElement.setAttribute("aria-label", "Interactive 3D dollhouse smart home. Drag to rotate and scroll to zoom.");
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.45, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 16;
    controls.maxDistance = 42;
    controls.maxPolarAngle = Math.PI * 0.46;
    controls.minPolarAngle = Math.PI * 0.12;
    controls.update();

    scene.add(new THREE.HemisphereLight(0xf4f5f1, 0x74776f, 2.0));
    const sun = new THREE.DirectionalLight(0xfff3dd, 3.2);
    sun.position.set(-8, 18, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -18; sun.shadow.camera.right = 18;
    sun.shadow.camera.top = 18; sun.shadow.camera.bottom = -18;
    sun.shadow.bias = -0.00025;
    scene.add(sun);

    const mats = {
      oak: new THREE.MeshStandardMaterial({ color: 0xb58a5e, roughness: 0.78 }),
      oakLight: new THREE.MeshStandardMaterial({ color: 0xd1ad80, roughness: 0.8 }),
      wall: new THREE.MeshStandardMaterial({ color: 0xe9e4d9, roughness: 0.88 }),
      wallSide: new THREE.MeshStandardMaterial({ color: 0xb8b7ae, roughness: 0.9 }),
      tile: new THREE.MeshStandardMaterial({ color: 0xcbd4d0, roughness: 0.42 }),
      bath: new THREE.MeshStandardMaterial({ color: 0xc3d8d9, roughness: 0.38 }),
      wood: new THREE.MeshStandardMaterial({ color: 0x79563b, roughness: 0.58 }),
      woodLight: new THREE.MeshStandardMaterial({ color: 0xb58b61, roughness: 0.58 }),
      fabric: new THREE.MeshStandardMaterial({ color: 0xb9b0a3, roughness: 0.96 }),
      fabricDark: new THREE.MeshStandardMaterial({ color: 0x777e78, roughness: 0.96 }),
      cream: new THREE.MeshStandardMaterial({ color: 0xf0e8d9, roughness: 0.95 }),
      metal: new THREE.MeshStandardMaterial({ color: 0x707a79, metalness: 0.62, roughness: 0.32 }),
      black: new THREE.MeshStandardMaterial({ color: 0x171c1b, roughness: 0.34 }),
      glass: new THREE.MeshStandardMaterial({ color: state("window") ? 0x91d7df : 0x90b9c1, emissive: state("window") ? 0x21515a : 0x081719, emissiveIntensity: state("window") ? 0.6 : 0.12, roughness: 0.16, metalness: 0.15, transparent: true, opacity: 0.9 }),
      green: new THREE.MeshStandardMaterial({ color: 0x526f59, roughness: 0.9 }),
      leaf: new THREE.MeshStandardMaterial({ color: 0x47704c, roughness: 0.9 }),
    };
    const activeMat = (color, emissive = color) => new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: 0.8, roughness: 0.3, metalness: 0.12 });
    const screenMat = new THREE.MeshStandardMaterial({ color: state("tv") ? 0x1e5d47 : 0x111b1a, emissive: state("tv") ? 0x1a9b67 : 0x000000, emissiveIntensity: state("tv") ? 1.25 : 0.04, roughness: 0.18, metalness: 0.2 });
    const lampMat = new THREE.MeshStandardMaterial({ color: state("light") ? 0xffe28a : 0x827e6b, emissive: state("light") ? 0xffc84f : 0x000000, emissiveIntensity: state("light") ? 2.4 : 0.02, roughness: 0.25 });

    const addBox = (parent, x, y, z, w, h, d, material, opts = {}) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
      mesh.position.set(x, y, z);
      mesh.castShadow = opts.castShadow !== false;
      mesh.receiveShadow = true;
      if (opts.rotationY) mesh.rotation.y = opts.rotationY;
      parent.add(mesh);
      return mesh;
    };
    const addCylinder = (parent, x, y, z, rTop, rBottom, h, material, segments = 20) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBottom, h, segments), material);
      mesh.position.set(x, y, z); mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
    };
    const addSphere = (parent, x, y, z, r, material) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 12), material);
      mesh.position.set(x, y, z); mesh.castShadow = true; parent.add(mesh); return mesh;
    };
    // Room names are rendered flat on the floor (not floating billboards), so the
    // camera never shows stray, oversized or garbled-looking characters above rooms.
    const addFloorLabel = (text, x, z, w = 2.2) => {
      const c = document.createElement("canvas"); c.width = 512; c.height = 128;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "rgba(248,246,239,.88)"; ctx.beginPath(); ctx.roundRect(10, 10, 492, 108, 18); ctx.fill();
      ctx.strokeStyle = "rgba(77,86,77,.28)"; ctx.lineWidth = 3; ctx.stroke();
      ctx.fillStyle = "#343c36"; ctx.font = "bold 36px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text.toUpperCase(), 256, 64);
      const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, w * 0.25), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      mesh.rotation.x = -Math.PI / 2; mesh.position.set(x, 0.108, z); scene.add(mesh); return mesh;
    };

    // Low architectural plinth and the five connected room zones.
    addBox(scene, 0, -0.24, 0, 16.8, 0.45, 11.2, new THREE.MeshStandardMaterial({ color: 0x5b605a, roughness: 0.82 }));
    const rooms = [
      { name: "Living Room", x: -4.15, z: -2.55, w: 7.8, d: 5.0, material: mats.oak },
      { name: "Bedroom", x: 4.05, z: -2.55, w: 7.7, d: 5.0, material: mats.oakLight },
      { name: "Kitchen", x: -4.15, z: 2.75, w: 7.8, d: 4.5, material: mats.tile },
      { name: "Bathroom", x: 1.15, z: 3.45, w: 2.55, d: 3.15, material: mats.bath },
      { name: "Hall / Entrance", x: 5.05, z: 2.75, w: 4.4, d: 4.5, material: mats.oakLight },
    ];
    rooms.forEach(r => {
      addBox(scene, r.x, 0, r.z, r.w, 0.18, r.d, r.material, { castShadow: false });
      // Fine floorboard / tile seams.
      if (r.material === mats.tile || r.material === mats.bath) {
        const seamMat = new THREE.MeshStandardMaterial({ color: 0xaabbb8, roughness: 0.85 });
        for (let x = r.x - r.w / 2 + 0.35; x < r.x + r.w / 2; x += 0.7) addBox(scene, x, 0.096, r.z, 0.012, 0.006, r.d - 0.05, seamMat, { castShadow: false });
        for (let z = r.z - r.d / 2 + 0.35; z < r.z + r.d / 2; z += 0.7) addBox(scene, r.x, 0.097, z, r.w - 0.05, 0.006, 0.012, seamMat, { castShadow: false });
      } else {
        const seamMat = new THREE.MeshStandardMaterial({ color: 0x98714b, roughness: 0.85 });
        for (let z = r.z - r.d / 2 + 0.42; z < r.z + r.d / 2; z += 0.42) addBox(scene, r.x, 0.096, z, r.w - 0.08, 0.006, 0.012, seamMat, { castShadow: false });
      }
      addFloorLabel(r.name, r.x, r.z, Math.min(2.5, r.w * 0.55));
    });

    // Back walls are tall; the front perimeter is cut low for the dollhouse view.
    const wallH = 1.72, lowH = 0.72, t = 0.22;
    const wallSeg = (x,z,w,d,h=wallH,mat=mats.wall) => addBox(scene,x,h/2,z,w,h,d,mat);
    // Exterior shell: one bedroom window opening and one front entrance opening.
    // Side walls are continuous, so there are no accidental black gaps in the shell.
    wallSeg(-8.3, 0, t, 11.0, wallH, mats.wallSide);
    // Back wall spans the living room completely; only the bedroom section is cut for a window.
    wallSeg(-6.625, -5.5, 3.35, t, wallH, mats.wall);
    wallSeg(-3.9, -5.5, 2.1, t, wallH, mats.wall); // closes the former living-room window gap
    wallSeg(0.15, -5.5, 6.0, t, wallH, mats.wall);
    wallSeg(6.775, -5.5, 3.05, t, wallH, mats.wall);
    wallSeg(8.3, 0, t, 11.0, wallH, mats.wallSide);
    // Front cutaway wall has one and only one entrance opening, aligned to the door.
    wallSeg(-4.65, 5.45, 7.3, t, lowH, mats.wallSide);
    wallSeg(1.375, 5.45, 3.95, t, lowH, mats.wallSide);
    wallSeg(6.625, 5.45, 3.35, t, lowH, mats.wallSide);
    wallSeg(-0.8, 5.45, 0.4, t, lowH, mats.wallSide); // close the small leftover gap
    // Low internal partitions keep room boundaries visible without hiding furniture.
    wallSeg(0, -2.55, t, 5.0, 1.02, mats.wallSide);
    wallSeg(-4.15, 0.02, 7.8, t, 0.92, mats.wallSide);
    wallSeg(4.05, 0.02, 7.7, t, 0.92, mats.wallSide);
    wallSeg(-0.05, 2.75, t, 4.5, 0.86, mats.wallSide);
    wallSeg(2.55, 3.45, t, 3.15, 0.9, mats.wallSide);
    wallSeg(1.15, 1.88, 2.55, t, 0.9, mats.wallSide);

    // Window frames with glazed panes on the exterior walls; all respond to the Window command.
    const windowGroup = new THREE.Group(); scene.add(windowGroup);
    const addWindow = (x,y,z,w,h,rot=0) => {
      const g = new THREE.Group(); g.position.set(x,y,z); g.rotation.y = rot; windowGroup.add(g);
      // Frame is built from four rails, never a solid board behind the glazing.
      const rail = 0.085;
      addBox(g, -w/2 + rail/2, 0, 0, rail, h, 0.12, mats.woodLight);
      addBox(g,  w/2 - rail/2, 0, 0, rail, h, 0.12, mats.woodLight);
      addBox(g, 0, -h/2 + rail/2, 0, w, rail, 0.12, mats.woodLight);
      addBox(g, 0,  h/2 - rail/2, 0, w, rail, 0.12, mats.woodLight);
      addBox(g, 0, 0, 0.025, 0.045, h-rail*2, 0.1, mats.cream, {castShadow:false});
      if (state("window")) {
        // Open casements swing away from the centre and leave a genuine open aperture.
        const left = addBox(g, -w*0.25, 0, 0.065, w*0.46, h-rail*2, 0.035, mats.glass, {castShadow:false});
        const right = addBox(g, w*0.25, 0, 0.065, w*0.46, h-rail*2, 0.035, mats.glass, {castShadow:false});
        left.rotation.y = -0.42; right.rotation.y = 0.42;
        left.position.z = 0.12; right.position.z = 0.12;
      } else {
        addBox(g,0,0,0.055,w-rail*2,h-rail*2,0.025,mats.glass,{castShadow:false});
      }
    };
    // The only window in the whole house is in the bedroom.
    addWindow(4.2, 1.3, -5.36, 2.1, 0.92);

    // Main front door, hinged so the accepted Door command visibly opens it.
    const doorFrame = new THREE.Group(); doorFrame.position.set(4.15, 0.08, 5.3); scene.add(doorFrame);
    // Jambs and lintel only: do not place a solid block behind the door leaf.
    addBox(doorFrame,-0.57,0.98,0,0.12,1.95,0.18,mats.woodLight);
    addBox(doorFrame, 0.57,0.98,0,0.12,1.95,0.18,mats.woodLight);
    addBox(doorFrame,0,1.94,0,1.26,0.12,0.18,mats.woodLight);
    const doorLeaf = new THREE.Group(); doorLeaf.position.set(-0.52,0.08,0.13); doorFrame.add(doorLeaf);
    addBox(doorLeaf,0.52,0.88,0,1.0,1.78,0.11,mats.wood);
    addBox(doorLeaf,0.52,0.9,0.065,0.72,1.35,0.035,mats.woodLight,{castShadow:false});
    addSphere(doorLeaf,0.88,0.86,0.105,0.055,mats.metal);
    doorLeaf.rotation.y = state("door") ? -Math.PI * 0.42 : 0;
    // Entrance threshold and exterior paving continue beyond the cutaway so the open
    // doorway looks outdoors rather than into a black void.
    addBox(scene,4.15,0.02,5.85,1.65,0.12,0.72,mats.wallSide);
    addBox(scene,4.15,-0.005,6.72,2.7,0.08,1.65,mats.tile,{castShadow:false});
    for (let i=0;i<4;i++) addBox(scene,3.05+i*0.73,0.04,6.72,0.012,0.006,1.55,new THREE.MeshStandardMaterial({color:0xaabbb8,roughness:0.85}),{castShadow:false});

    // Living room: sofa, cushions, coffee table, TV console, TV, telephone, floor lamp and radio.
    const living = new THREE.Group(); scene.add(living);
    addBox(living,-5.3,0.42,-1.15,3.05,0.72,1.0,mats.fabric);
    addBox(living,-5.3,0.82,-1.63,3.05,0.55,0.22,mats.fabricDark);
    [-6.25,-5.3,-4.35].forEach(x=>addBox(living,x,0.86,-1.05,0.76,0.2,0.68,mats.cream));
    addBox(living,-5.3,0.27,-1.78,3.12,0.25,0.2,mats.wood);
    addBox(living,-3.65,0.25,-2.75,2.1,0.18,1.18,mats.woodLight);
    [-4.48,-2.82].forEach(x=>[-3.18,-2.32].forEach(z=>addBox(living,x,0.12,z,0.09,0.24,0.09,mats.wood)));
    addBox(living,-1.3,0.38,-4.42,2.7,0.5,0.62,mats.wood);
    addBox(living,-1.3,1.0,-4.42,2.45,0.88,0.12,mats.black);
    addBox(living,-1.3,1.02,-4.34,2.28,0.68,0.035,screenMat,{castShadow:false});
    addBox(living,-1.3,0.13,-4.42,0.6,0.18,0.6,mats.wood);
    // Telephone on a side table.
    addBox(living,-6.65,0.42,-3.6,0.78,0.68,0.7,mats.woodLight);
    addBox(living,-6.65,0.79,-3.6,0.72,0.08,0.64,mats.wood);
    addBox(living,-6.65,0.91,-3.6,0.42,0.13,0.29,state("telephone")?activeMat(0x315d4b,0x1c8b59):mats.black);
    addBox(living,-6.65,1.01,-3.6,0.48,0.08,0.1,mats.metal);
    // Radio with tuning knobs and antenna.
    addBox(living,-2.25,0.43,-3.25,0.75,0.48,0.38,state("radio")?activeMat(0x416d56,0x18834c):mats.wood);
    addBox(living,-2.25,0.46,-3.04,0.44,0.24,0.035,mats.black,{castShadow:false});
    for(let i=0;i<5;i++) addBox(living,-2.42+i*0.085,0.46,-3.015,0.035,0.17,0.018,state("radio")?activeMat(0x7ce5a3,0x4abf79):mats.metal,{castShadow:false});
    addCylinder(living,-2.5,0.78,-3.3,0.018,0.018,0.6,mats.metal,8).rotation.z=-0.35;
    // Floor lamp.
    addCylinder(living,-7.1,0.8,-0.1,0.045,0.045,1.5,mats.metal,12);
    addCylinder(living,-7.1,1.62,-0.1,0.34,0.19,0.42,lampMat,20);

    // Bedroom: realistic bed, bedside tables, wardrobe and a window-side plant.
    const bed = new THREE.Group(); scene.add(bed);
    addBox(bed,3.75,0.27,-2.5,3.15,0.42,3.45,mats.wood);
    addBox(bed,3.75,0.53,-2.5,2.95,0.22,3.25,mats.cream);
    addBox(bed,3.75,0.67,-2.92,2.8,0.18,2.1,new THREE.MeshStandardMaterial({color:0x9caeae,roughness:0.96}));
    addBox(bed,3.0,0.79,-3.75,1.0,0.22,0.58,mats.cream);
    addBox(bed,4.48,0.79,-3.75,1.0,0.22,0.58,mats.cream);
    addBox(bed,3.8,0.52,-4.27,3.3,0.88,0.16,mats.woodLight);
    addBox(bed,6.7,0.5,-4.25,1.15,0.8,0.68,mats.woodLight);
    addBox(bed,6.7,0.94,-4.25,1.18,0.08,0.7,mats.wood);
    addBox(bed,6.7,1.04,-4.25,0.44,0.1,0.26,lampMat);
    addBox(bed,1.5,0.42,-4.3,0.9,0.7,0.65,mats.woodLight);
    addBox(bed,1.5,0.8,-4.3,0.94,0.08,0.69,mats.wood);

    // Kitchen: counter run, lower cabinets, sink, cooktop, refrigerator and island.
    const kitchen = new THREE.Group(); scene.add(kitchen);
    addBox(kitchen,-6.7,0.46,4.28,2.55,0.82,0.72,mats.woodLight);
    addBox(kitchen,-6.7,0.91,4.28,2.7,0.12,0.82,mats.cream);
    addBox(kitchen,-6.7,1.42,4.28,2.55,0.82,0.56,mats.cream);
    addBox(kitchen,-6.7,1.43,3.97,2.35,0.025,0.04,mats.metal,{castShadow:false});
    addBox(kitchen,-6.1,1.0,4.24,0.65,0.04,0.43,mats.metal,{castShadow:false});
    addBox(kitchen,-4.4,0.92,4.22,0.74,0.04,0.58,mats.black,{castShadow:false});
    [0,1].forEach(i=>[0,1].forEach(j=>addCylinder(kitchen,-4.62+i*0.42,0.95,4.04+j*0.34,0.095,0.095,0.025,mats.metal,18)));
    addBox(kitchen,-2.2,0.8,4.15,0.98,1.55,0.9,mats.cream);
    addBox(kitchen,-2.2,1.62,4.15,1.02,0.08,0.94,mats.metal);
    addBox(kitchen,-4.4,0.44,2.6,2.1,0.78,0.9,mats.woodLight);
    addBox(kitchen,-4.4,0.86,2.6,2.18,0.1,0.96,mats.cream);
    [-5.1,-4.4,-3.7].forEach(x=>addBox(kitchen,x,0.45,2.13,0.035,0.64,0.02,mats.wood));
    // Small dining table and chairs.
    addBox(kitchen,-5.0,0.62,0.95,1.8,0.14,1.1,mats.wood);
    [-5.65,-4.35].forEach(x=>[-0.0,1.9].forEach(z=>{addBox(kitchen,x,0.34,z,0.48,0.12,0.45,mats.fabricDark); addBox(kitchen,x,0.18,z,0.42,0.3,0.4,mats.wood);}));

    // Compact bathroom: toilet, vanity basin and glass shower enclosure.
    const bath = new THREE.Group(); scene.add(bath);
    addBox(bath,0.5,0.36,4.25,0.82,0.65,0.65,mats.cream);
    addSphere(bath,0.5,0.72,4.25,0.29,mats.cream).scale.set(1,0.42,1);
    addBox(bath,0.5,0.78,3.98,0.52,0.06,0.08,mats.cream);
    addBox(bath,1.95,0.48,2.48,0.88,0.82,0.62,mats.woodLight);
    addBox(bath,1.95,0.91,2.48,0.96,0.12,0.7,mats.cream);
    addSphere(bath,1.95,0.99,2.48,0.23,mats.cream).scale.set(1.2,0.35,0.8);
    addBox(bath,1.95,0.95,3.1,0.9,1.65,0.035,new THREE.MeshStandardMaterial({color:0x9acbd0,transparent:true,opacity:0.34,roughness:0.12,metalness:0.1}),{castShadow:false});
    addBox(bath,1.95,0.12,3.1,0.94,0.12,0.92,mats.cream);

    // Hall console, shoe cabinet and potted plant.
    const hall = new THREE.Group(); scene.add(hall);
    addBox(hall,6.45,0.42,2.1,1.75,0.72,0.62,mats.woodLight);
    addBox(hall,6.45,0.81,2.1,1.8,0.08,0.66,mats.wood);
    addBox(hall,6.45,1.08,2.1,0.62,0.42,0.045,mats.black);
    addCylinder(hall,7.3,0.45,4.2,0.28,0.25,0.55,mats.cream,18);
    addCylinder(hall,7.3,0.92,4.2,0.08,0.1,0.58,mats.wood,12);
    for(let i=0;i<5;i++) addSphere(hall,7.3+Math.cos(i*1.25)*0.22,1.25+(i%2)*0.12,4.2+Math.sin(i*1.25)*0.22,0.22,mats.leaf);

    // Each room gets a practical ceiling fixture and point light; the generic Light command lights all rooms.
    const roomLights = [];
    const addCeilingLight = (x,z) => {
      addCylinder(scene,x,2.15,z,0.28,0.32,0.1,mats.metal,18);
      addSphere(scene,x,2.06,z,0.18,lampMat);
      const light = new THREE.PointLight(0xffd982, state("light") ? 1.65 : 0.03, 7.5, 1.7);
      light.position.set(x,1.85,z); scene.add(light); roomLights.push(light);
    };
    // Only the living room and hallway have visible ceiling lights, per the requested layout.
    addCeilingLight(-4.2,-2.55);
    addCeilingLight(5.3,2.8);
    // Green status pips on the appliances that have a direct command mapping.
    const statusPip = (x,y,z,on) => addSphere(scene,x,y,z,0.085,on?activeMat(0x6ff0a5,0x35e984):new THREE.MeshStandardMaterial({color:0x606964,roughness:0.5}));
    statusPip(-1.3,1.56,-4.23,state("tv"));
    statusPip(-6.65,1.15,-3.6,state("telephone"));
    statusPip(-2.25,0.77,-3.0,state("radio"));
    statusPip(4.8,1.5,5.1,state("door"));

    // Subtle ground shadow makes the model feel like an architectural miniature.
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200,200), new THREE.ShadowMaterial({opacity:0.16}));
    ground.rotation.x = -Math.PI/2; ground.position.y = -0.48; ground.receiveShadow = true; scene.add(ground);

    const resize = () => {
      const width = Math.max(host.clientWidth, 320);
      const height = Math.max(host.clientHeight, 420);
      camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width,height,false);
    };
    resize();
    const observer = new ResizeObserver(resize); observer.observe(host);
    let frame = 0;
    const animate = () => { frame = requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); };
    animate();

    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
      scene.traverse(obj => { if (obj.geometry) obj.geometry.dispose(); if (obj.material) { const materials = Array.isArray(obj.material) ? obj.material : [obj.material]; materials.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); }); } });
      renderer.dispose(); if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement);
    };
  }, [devices]);

  const appliance = (id, label, icon, openClose = false) => (
    <div className={`floor-appliance ${state(id) ? "is-active" : ""}`} key={id}>
      <span className="floor-appliance-icon" aria-hidden="true">{icon}</span>
      <span className="floor-appliance-name">{label}</span>
      <strong>{openClose ? (state(id) ? "OPEN" : "CLOSED") : (state(id) ? "ON" : "OFF")}</strong>
    </div>
  );

  return (
    <section className="whole-home-panel" aria-label="Interactive 3D whole home model">
      <div className="whole-home-heading">
        <div>
          <div className="eyebrow">LIVE 3D SMART HOME</div>
          <h2>Interactive 3D Dollhouse</h2>
          <p>Whole furnished home · drag to rotate · scroll to zoom · device states follow the executed EEG command.</p>
        </div>
        <span className="floor-live-badge"><i /> LIVE DEVICE STATES</span>
      </div>
      <div className="neurohome-3d-stage" ref={mountRef} />
      <div className="neurohome-3d-hint">LIVING ROOM · BEDROOM (ONE WINDOW) · KITCHEN · COMPACT BATHROOM · HALLWAY · ONE FRONT ENTRANCE</div>
      <div className="floor-appliance-summary">
        {appliance("tv", "Television · Living Room", "▣")}
        {appliance("telephone", "Telephone · Living Room", "☎")}
        {appliance("light", "Light · Whole Home", "☼")}
        {appliance("door", "Front Door · Entrance", "▤", true)}
        {appliance("window", "Bedroom Window", "▥", true)}
        {appliance("radio", "Radio · Living Room", "♫")}
      </div>
      <p className="floor-plan-note">The 3D model visualizes the home and its device states. Light illuminates the living room and hallway; Window opens or closes the single bedroom window. The P300 model selects an appliance command and does not infer physical location.</p>
    </section>
  );
}

function DeviceVisual({ type, on }) {
  if (type === "light") {
    return (
      <div className={`deviceVisual bulbVisual ${on ? "active" : ""}`}>
        <div className="bulb">
          <span />
        </div>
        <div className="bulbBase" />
      </div>
    );
  }

  if (type === "tv") {
    return (
      <div className={`deviceVisual tvVisual ${on ? "active" : ""}`}>
        <div className="tvScreen">
          <span>{on ? "LIVE" : "OFF"}</span>
        </div>
        <div className="tvStand" />
      </div>
    );
  }

  if (type === "telephone") {
    return (
      <div
        className={`deviceVisual telephoneVisual ${
          on ? "active" : ""
        }`}
      >
        <div className="phoneBody">
          <span>☎</span>
        </div>

        {on && (
          <div className="callWaves">
            <i />
            <i />
            <i />
          </div>
        )}
      </div>
    );
  }

  if (type === "window") {
    return (
      <div
        className={`deviceVisual windowVisual ${
          on ? "active" : ""
        }`}
      >
        <div className="windowFrame">
          <i />
          <i />
          <i />
          <i />
        </div>
      </div>
    );
  }

  if (type === "radio") {
    return (
      <div
        className={`deviceVisual radioVisual ${
          on ? "active" : ""
        }`}
      >
        <div className="radioBox">
          <span className="radioDial" />
          <span className="radioLabel">RADIO</span>

          <div className="songSignal">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`deviceVisual doorVisual ${
        on ? "open" : "closed"
      }`}
    >
      <div className="doorFrame">
        <div className="doorLeaf">
          <span />
        </div>
      </div>
    </div>
  );
}

function ModelPage() {
  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">MODEL RESULTS</div>

        <h1>EEGNet performance.</h1>

        <p>
          Final evaluation on the untouched Run 5. These are
          the model metrics integrated into this prototype.
        </p>
      </section>

      <section className="metricsLarge">
        <Metric title="Accuracy" value={MODEL.accuracy} />
        <Metric
          title="Balanced Accuracy"
          value={MODEL.balanced}
        />
        <Metric title="Target F1" value={MODEL.targetF1} />
      </section>

      <div className="twoCol">
        <Panel title="MODEL CONFIGURATION">
          <div className="details">
            <Info label="ARCHITECTURE" value="EEGNet" />
            <Info label="INPUT" value="32 x 154 x 1" />
            <Info label="WINDOW" value={MODEL.window} />
            <Info label="THRESHOLD" value={MODEL.threshold} />
            <Info label="TRAINING RUNS" value="0-4" />
            <Info label="TEST RUN" value="5" />
          </div>
        </Panel>

        <Panel title="CONFUSION MATRIX">
          <div className="matrix">
            <div>TRUE / PRED</div>
            <b>NonTarget</b>
            <b>Target</b>
            <b>NonTarget</b>
            <strong>86</strong>
            <strong>36</strong>
            <b>Target</b>
            <strong>8</strong>
            <strong>17</strong>
          </div>
        </Panel>
      </div>
    </>
  );
}

function ResultSummary({ data }) {
  return (
    <div className="resultSummary">
      <span>ANALYSIS COMPLETE</span>
      <strong>{data?.epochs ?? 0} epochs</strong>
      <b>{data?.predicted_target_count ?? 0} Target</b>
      <b>{data?.predicted_nontarget_count ?? 0} Non-Target</b>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{sub}</small>
    </div>
  );
}

function Metric({ title, value }) {
  const width = Number.parseFloat(value) || 0;

  return (
    <div className="metric">
      <span>{title}</span>
      <strong>{value}</strong>

      <div className="meter">
        <i style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="info">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Flow({ n, t }) {
  return (
    <div className="flowItem">
      <small>{n}</small>
      <strong>{t}</strong>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <div className="panelTitle">{title}</div>
      {children}
    </section>
  );
}


