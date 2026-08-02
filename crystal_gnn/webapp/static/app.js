const sampleCif = `data_NaCl
_symmetry_space_group_name_H-M 'F m -3 m'
_cell_length_a 5.6402
_cell_length_b 5.6402
_cell_length_c 5.6402
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 225
_chemical_formula_structural NaCl
_chemical_formula_sum 'Na1 Cl1'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5`;

const statusEl = document.getElementById("status");
const inputEl = document.getElementById("structure-input");
const checkpointEl = document.getElementById("checkpoint");
const fmtEl = document.getElementById("fmt");
const mcEl = document.getElementById("mc");

const metricEls = {
  pred: document.getElementById("pred"),
  interval: document.getElementById("interval"),
  ale: document.getElementById("ale"),
  epi: document.getElementById("epi"),
  tot: document.getElementById("tot"),
};

let crystalScene;
let crystalRenderer;
let crystalCamera;
let crystalControls;
let crystalRoot;
let graphRenderer;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b1163a" : "#425b84";
}

function formatN(value, digits = 6) {
  if (!Number.isFinite(value)) return "-";
  return Number(value).toFixed(digits);
}

function elementColor(symbol) {
  const code = [...symbol].reduce((acc, c) => acc + c.charCodeAt(0), 0);
  const hue = code % 360;
  return new THREE.Color(`hsl(${hue}, 75%, 52%)`);
}

function clearCrystalScene() {
  if (crystalRoot) {
    crystalScene.remove(crystalRoot);
  }
  crystalRoot = new THREE.Group();
  crystalScene.add(crystalRoot);
}

function addLatticeLines(lattice) {
  const v = lattice.map((row) => new THREE.Vector3(row[0], row[1], row[2]));
  const origin = new THREE.Vector3(0, 0, 0);
  const corners = [
    origin,
    v[0],
    v[1],
    v[2],
    v[0].clone().add(v[1]),
    v[0].clone().add(v[2]),
    v[1].clone().add(v[2]),
    v[0].clone().add(v[1]).add(v[2]),
  ];

  const idx = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [1, 5],
    [2, 4],
    [2, 6],
    [3, 5],
    [3, 6],
    [4, 7],
    [5, 7],
    [6, 7],
  ];

  const mat = new THREE.LineBasicMaterial({
    color: 0x1f4a8b,
    transparent: true,
    opacity: 0.8,
  });
  idx.forEach(([a, b]) => {
    const geom = new THREE.BufferGeometry().setFromPoints([
      corners[a],
      corners[b],
    ]);
    const line = new THREE.Line(geom, mat);
    crystalRoot.add(line);
  });
}

function renderCrystal(crystal, graphLinks) {
  clearCrystalScene();
  addLatticeLines(crystal.lattice);

  crystal.nodes.forEach((node) => {
    const geo = new THREE.SphereGeometry(0.25, 18, 18);
    const mat = new THREE.MeshStandardMaterial({
      color: elementColor(node.element),
      roughness: 0.3,
      metalness: 0.15,
    });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(node.cart[0], node.cart[1], node.cart[2]);
    crystalRoot.add(sphere);
  });

  const dedup = new Set();
  graphLinks.forEach((e) => {
    const a = Math.min(e.source, e.target);
    const b = Math.max(e.source, e.target);
    const key = `${a}-${b}`;
    if (a === b || dedup.has(key)) return;
    dedup.add(key);

    const pa = new THREE.Vector3(...crystal.nodes[a].cart);
    const pb = new THREE.Vector3(...crystal.nodes[b].cart);
    const geom = new THREE.BufferGeometry().setFromPoints([pa, pb]);
    const line = new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({
        color: 0x1b3b6f,
        transparent: true,
        opacity: 0.35,
      }),
    );
    crystalRoot.add(line);
  });

  const box = new THREE.Box3().setFromObject(crystalRoot);
  const center = box.getCenter(new THREE.Vector3());
  crystalRoot.position.sub(center);

  const size = box.getSize(new THREE.Vector3()).length() || 5;
  crystalCamera.position.set(size * 0.8, size * 0.8, size * 1.15);
  crystalControls.update();
}

function renderGraph(graph) {
  const data = {
    nodes: graph.nodes.map((n) => ({
      id: n.id,
      name: n.name,
      color: `#${elementColor(n.name).getHexString()}`,
    })),
    links: graph.links.map((e) => ({
      source: e.source,
      target: e.target,
      value: 1 / Math.max(e.distance, 1e-6),
    })),
  };

  graphRenderer
    .graphData(data)
    .nodeLabel((n) => `${n.name} (#${n.id})`)
    .nodeRelSize(5)
    .nodeResolution(12)
    .backgroundColor("rgba(0,0,0,0)")
    .linkOpacity(0.45)
    .linkWidth((l) => Math.min(2, 0.4 + l.value));
}

function initCrystalScene() {
  const host = document.getElementById("crystal-view");
  crystalScene = new THREE.Scene();
  crystalScene.background = new THREE.Color(0xf5f9ff);

  const w = host.clientWidth;
  const h = host.clientHeight;
  crystalCamera = new THREE.PerspectiveCamera(50, w / h, 0.01, 1000);
  crystalCamera.position.set(7, 6, 8);

  crystalRenderer = new THREE.WebGLRenderer({ antialias: true });
  crystalRenderer.setSize(w, h);
  host.innerHTML = "";
  host.appendChild(crystalRenderer.domElement);

  crystalControls = new THREE.OrbitControls(
    crystalCamera,
    crystalRenderer.domElement,
  );
  crystalControls.enableDamping = true;

  const key = new THREE.DirectionalLight(0xffffff, 1.0);
  key.position.set(8, 10, 9);
  crystalScene.add(key);
  crystalScene.add(new THREE.AmbientLight(0xffffff, 0.6));

  clearCrystalScene();
}

function initGraphScene() {
  const host = document.getElementById("graph-view");
  graphRenderer = ForceGraph3D()(host)
    .backgroundColor("rgba(0,0,0,0)")
    .linkDirectionalParticles(0);
}

function animate() {
  requestAnimationFrame(animate);
  crystalControls.update();
  crystalRenderer.render(crystalScene, crystalCamera);
}

async function fetchDefaultCheckpoint() {
  try {
    const r = await fetch("/api/checkpoint/default");
    if (!r.ok) return;
    const data = await r.json();
    checkpointEl.value = data.checkpoint;
  } catch (_) {
    // Keep manual checkpoint fallback.
  }
}

async function runPrediction() {
  const structureText = inputEl.value.trim();
  if (!structureText) {
    setStatus("Please paste a crystal structure first.", true);
    return;
  }

  const payload = {
    structure_text: structureText,
    structure_format: fmtEl.value,
    checkpoint: checkpointEl.value.trim() || null,
    mc_samples: Number(mcEl.value) || 10,
  };

  setStatus("Running model inference and generating 3D visuals...");

  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) {
      throw new Error(data.detail || "Prediction request failed");
    }

    metricEls.pred.textContent = `${formatN(data.prediction, 6)} eV/atom`;
    metricEls.interval.textContent = `[${formatN(data.interval_90[0], 4)}, ${formatN(data.interval_90[1], 4)}]`;
    metricEls.ale.textContent = formatN(data.aleatoric_uncertainty, 6);
    metricEls.epi.textContent = formatN(data.epistemic_uncertainty, 6);
    metricEls.tot.textContent = formatN(data.total_uncertainty, 6);

    renderCrystal(data.crystal, data.graph.links);
    renderGraph(data.graph);

    setStatus(
      `Done. Checkpoint: ${data.checkpoint} | Graph radius: ${data.graph.radius}`,
    );
  } catch (err) {
    setStatus(err.message || "Unknown error", true);
  }
}

window.addEventListener("resize", () => {
  const host = document.getElementById("crystal-view");
  const w = host.clientWidth;
  const h = host.clientHeight;
  crystalCamera.aspect = w / h;
  crystalCamera.updateProjectionMatrix();
  crystalRenderer.setSize(w, h);
});

document.getElementById("predict-btn").addEventListener("click", runPrediction);
document.getElementById("sample-btn").addEventListener("click", () => {
  inputEl.value = sampleCif;
  setStatus("Sample CIF loaded.");
});

initCrystalScene();
initGraphScene();
animate();
fetchDefaultCheckpoint();
inputEl.value = sampleCif;
