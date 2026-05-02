const APP_CONFIG = {
  previewRows: 7,
};

const BASE_URL = "http://localhost:8000";

let uploadedFile = null;
let columns = [];
let uploadedDataset = null;
let lastTrainingRun = null;

const taskTypeSelect = document.getElementById("taskType");
const targetDiv = document.getElementById("targetDiv");
const trainBtn = document.getElementById("trainBtn");
const downloadBtn = document.getElementById("downloadBtn");
const previewPanel = document.getElementById("preview");
const resultsPanel = document.getElementById("results");

//task type selection
taskTypeSelect.addEventListener("change", function () {
  const requiresTarget =
    this.value === "classification" || this.value === "regression";
  const isClustering = this.value === "clustering";

  targetDiv.classList.toggle("is-hidden", !requiresTarget);
  document
    .getElementById("clustersDiv")
    .classList.toggle("is-hidden", !isClustering);

  if (requiresTarget) {
    loadColumns();
  }
});

//status
function setStatus(message, type = "info") {
  const status = document.getElementById("statusMessage");
  status.textContent = message;
  status.className = `status-banner ${type}`;
}
//button label helpers
function setButtonLabel(button, label) {
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.textContent;
  }
  button.textContent = label;
}
//reset button to original label
function resetButtonLabel(button) {
  if (button.dataset.defaultLabel) {
    button.textContent = button.dataset.defaultLabel;
  }
}
//empty state
function setEmptyState(element, message) {
  element.classList.add("empty-state");
  element.textContent = message;
}
//format values for display
function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }

  return String(value);
}
//render dataset preview
function renderPreview(dataset) {
  previewPanel.classList.remove("empty-state");
  previewPanel.innerHTML = "";

  const stack = document.createElement("div");
  stack.className = "panel-stack";

  const metaGrid = document.createElement("div");
  metaGrid.className = "dataset-meta";

  [
    { label: "Source", value: dataset.source },
    { label: "Rows Detected", value: formatValue(dataset.rowCount) },
    { label: "Columns", value: formatValue(dataset.columnCount) },
  ].forEach((item) => {
    const tile = document.createElement("div");
    tile.className = "meta-tile";

    const label = document.createElement("span");
    label.className = "meta-label";
    label.textContent = item.label;

    const value = document.createElement("span");
    value.className = "meta-value";
    value.textContent = item.value;

    tile.appendChild(label);
    tile.appendChild(value);
    metaGrid.appendChild(tile);
  });

  stack.appendChild(metaGrid);

  if (
    Array.isArray(dataset.preview) &&
    dataset.preview.length > 0 &&
    typeof dataset.preview[0] === "object" &&
    dataset.preview[0] !== null &&
    !Array.isArray(dataset.preview[0])
  ) {
    const tableWrap = document.createElement("div");
    tableWrap.className = "table-wrap";

    const table = document.createElement("table");
    table.className = "data-table";

    const previewColumns = Object.keys(dataset.preview[0]);
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    previewColumns.forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column;
      headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    dataset.preview.forEach((row) => {
      const tr = document.createElement("tr");

      previewColumns.forEach((column) => {
        const td = document.createElement("td");
        td.textContent = formatValue(row[column]);
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    tableWrap.appendChild(table);
    stack.appendChild(tableWrap);
  } else {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(dataset.preview, null, 2);
    stack.appendChild(pre);
  }

  previewPanel.appendChild(stack);
}
//load columns into target column dropdown
function loadColumns() {
  const select = document.getElementById("targetColumn");
  select.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = columns.length
    ? "Select target column"
    : "No columns available";
  placeholder.disabled = true;
  placeholder.selected = true;
  select.appendChild(placeholder);

  columns.forEach((columnName) => {
    const option = document.createElement("option");
    option.value = columnName;
    option.textContent = columnName;
    select.appendChild(option);
  });
}
//render confusion matrix
function renderConfusionMatrix(matrix) {
  const wrap = document.createElement("div");
  wrap.className = "cm-wrap";

  const title = document.createElement("p");
  title.className = "cm-title";
  title.textContent = "Confusion Matrix";
  wrap.appendChild(title);

  const n = matrix.length;

  const table = document.createElement("table");
  table.className = "cm-table";

  // Header row
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const emptyTh = document.createElement("th");
  emptyTh.textContent = "Actual \\ Predicted";
  headerRow.appendChild(emptyTh);
  for (let j = 0; j < n; j++) {
    const th = document.createElement("th");
    th.textContent = "Class " + j;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body rows
  const tbody = document.createElement("tbody");
  const maxVal = Math.max(...matrix.flat());

  matrix.forEach((row, i) => {
    const tr = document.createElement("tr");

    const rowLabel = document.createElement("th");
    rowLabel.textContent = "Class " + i;
    tr.appendChild(rowLabel);

    row.forEach((val, j) => {
      const td = document.createElement("td");
      td.textContent = val;
      const intensity = maxVal > 0 ? val / maxVal : 0;
      if (i === j) {
        // Diagonal correct predictions (green color)
        td.style.background = `rgba(18, 113, 91, ${0.1 + intensity * 0.55})`;
        td.style.color = intensity > 0.5 ? "#0a4f3a" : "var(--text)";
        td.style.fontWeight = "700";
      } else if (val > 0) {
        // Off-diagonal errors (red color)
        td.style.background = `rgba(180, 35, 24, ${0.05 + intensity * 0.4})`;
        td.style.color = intensity > 0.5 ? "#7a1810" : "var(--text)";
      }
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  wrap.appendChild(table);

  const legend = document.createElement("p");
  legend.className = "cm-legend";
  legend.textContent = "Green = correct predictions · Red = misclassifications";
  wrap.appendChild(legend);

  return wrap;
}

// ── Optimal k silhouette chart ──────────────────────────────────────────
function renderKChart(optimalK, kScores) {
  const entries = Object.entries(kScores).map(([k, v]) => ({ k: parseInt(k), v }));
  entries.sort((a, b) => a.k - b.k);
  const bestScore = Math.max(...entries.map(e => e.v));

  const wrap = document.createElement("div");
  wrap.className = "kchart-wrap";

  // header
  wrap.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
      <p style="margin:0;font-size:11px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--color-text-secondary);">Optimal k selection</p>
      <span class="best-badge" style="font-size:12px;padding:3px 10px;">k = ${optimalK} selected</span>
    </div>
    <p style="margin:0 0 16px;font-size:13px;color:var(--color-text-secondary);">Silhouette score for each k tested (2–10). Highest score wins.</p>
    <div style="display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap;">
      <div class="kchart-stat">
        <span class="kchart-stat-label">Best k</span>
        <span class="kchart-stat-val">${optimalK}</span>
      </div>
      <div class="kchart-stat">
        <span class="kchart-stat-label">Best score</span>
        <span class="kchart-stat-val" style="color:var(--color-text-success);">${bestScore.toFixed(3)}</span>
      </div>
      <div class="kchart-stat">
        <span class="kchart-stat-label">k range</span>
        <span class="kchart-stat-val">2 – 10</span>
      </div>
    </div>
    <div style="position:relative;width:100%;height:200px;">
      <canvas id="kChartCanvas" role="img" aria-label="Bar chart of silhouette scores for k values from 2 to 10.">Silhouette scores per k value.</canvas>
    </div>
    <div style="display:flex;align-items:center;gap:16px;margin-top:12px;flex-wrap:wrap;">
      <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--color-text-secondary);">
        <span style="width:10px;height:10px;border-radius:2px;background:#1D9E75;display:inline-block;"></span>Best k
      </span>
      <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--color-text-secondary);">
        <span style="width:10px;height:10px;border-radius:2px;background:#B5D4F4;display:inline-block;"></span>Other k values
      </span>
    </div>
  `;

  // load Chart.js and draw after DOM insertion
  setTimeout(() => {
    if (!window.Chart) {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js";
      script.onload = () => drawKChart(entries, optimalK);
      document.head.appendChild(script);
    } else {
      drawKChart(entries, optimalK);
    }
  }, 100);

  return wrap;
}

function drawKChart(entries, optimalK) {
  const canvas = document.getElementById("kChartCanvas");
  if (!canvas) return;
  new Chart(canvas, {
    type: "bar",
    data: {
      labels: entries.map(e => `k=${e.k}`),
      datasets: [{
        label: "Silhouette score",
        data: entries.map(e => e.v),
        backgroundColor: entries.map(e => e.k === optimalK ? "#1D9E75" : "#B5D4F4"),
        borderColor:     entries.map(e => e.k === optimalK ? "#0F6E56" : "#378ADD"),
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `Silhouette: ${ctx.parsed.y.toFixed(3)}${ctx.label === "k="+optimalK ? " ★ best" : ""}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 12 }, autoSkip: false, maxRotation: 0 }
        },
        y: {
          min: 0,
          grid: { color: "rgba(0,0,0,0.06)" },
          ticks: { font: { size: 11 }, callback: v => v.toFixed(2) }
        }
      }
    }
  });
}


//render comparison of all models
function renderModelComparison(allModels, bestModel) {
  const section = document.createElement("div");
  section.className = "comparison-section";

  const title = document.createElement("p");
  title.className = "comparison-title";
  title.textContent = "All models comparison";
  section.appendChild(title);

  Object.entries(allModels).forEach(([modelName, metrics]) => {
    const card = document.createElement("div");
    card.className =
      "comparison-card" + (modelName === bestModel ? " is-best" : "");

    const header = document.createElement("div");
    header.className = "comparison-card-header";

    const name = document.createElement("span");
    name.className = "comparison-model-name";
    name.textContent = modelName;

    header.appendChild(name);

    if (modelName === bestModel) {
      const badge = document.createElement("span");
      badge.className = "best-badge";
      badge.textContent = "Best";
      header.appendChild(badge);
    }

    card.appendChild(header);

    const metricsGrid = document.createElement("div");
    metricsGrid.className = "comparison-metrics";

    Object.entries(metrics).forEach(([key, value]) => {
      if (key === "confusion_matrix") return; // shown separately
      const item = document.createElement("div");
      item.className = "comparison-metric-item";

      const k = document.createElement("span");
      k.className = "comparison-metric-key";
      k.textContent = key;

      const v = document.createElement("span");
      v.className = "comparison-metric-val";
      v.textContent = formatValue(value);

      item.appendChild(k);
      item.appendChild(v);
      metricsGrid.appendChild(item);
    });

    card.appendChild(metricsGrid);
    section.appendChild(card);
  });

  return section;
}
//display results in the UI
function displayResults(data) {
  const metrics = data?.metrics || {};
  const entries = Object.entries(metrics);

  resultsPanel.innerHTML = "";
  resultsPanel.classList.remove("empty-state", "results-grid");
  resultsPanel.classList.add("results-full");

  if (entries.length === 0) {
    resultsPanel.classList.add("empty-state");
    resultsPanel.classList.remove("results-full");
    resultsPanel.textContent =
      "Training finished, but no metrics were returned.";
    return;
  }

  //Best model banner
  if (data.best_model) {
    const banner = document.createElement("div");
    banner.className = "best-model-banner";
    banner.innerHTML = `<span class="best-model-label">Best model</span><span class="best-model-name">${data.best_model}</span>`;
    resultsPanel.appendChild(banner);
  }

  //Best model metrics cards
  const grid = document.createElement("div");
  grid.className = "results-grid-inner";

  let confusionMatrix = null;

  entries.forEach(([key, value]) => {
    if (key === "confusion_matrix") {
      confusionMatrix = value;
      return;
    }

    const card = document.createElement("div");
    card.className = "metric";

    const label = document.createElement("span");
    label.className = "metric-label";
    label.textContent = key;

    const metricValue = document.createElement("span");
    metricValue.className = "metric-value";
    metricValue.textContent = formatValue(value);

    card.appendChild(label);
    card.appendChild(metricValue);
    grid.appendChild(card);
  });

  resultsPanel.appendChild(grid);

  //Confusion matrix visual
  if (confusionMatrix && Array.isArray(confusionMatrix)) {
    resultsPanel.appendChild(renderConfusionMatrix(confusionMatrix));
  }

  // All models comparison
  if (
    data.all_models_metrics &&
    Object.keys(data.all_models_metrics).length > 1
  ) {
    resultsPanel.appendChild(
      renderModelComparison(data.all_models_metrics, data.best_model),
    );
  }

  // Optimal k chart (clustering only)
  if (data.optimal_k && data.k_scores) {
    resultsPanel.appendChild(renderKChart(data.optimal_k, data.k_scores));
  }
}

//upload file to backend
async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");

  if (!fileInput.files.length) {
    setStatus("Choose a CSV or Excel file before uploading.", "error");
    return;
  }

  uploadedFile = fileInput.files[0];
  lastTrainingRun = null;
  downloadBtn.disabled = true;
  trainBtn.disabled = true;
  uploadBtn.disabled = true;

  setButtonLabel(uploadBtn, "Uploading...");
  setStatus(`Uploading ${uploadedFile.name} to the backend...`, "info");
  setEmptyState(resultsPanel, "Run AutoML to display evaluation metrics.");

  try {
    const formData = new FormData();
    formData.append("file", uploadedFile);

    const res = await fetch(`${BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed.");
    }

    const data = await res.json();

    uploadedDataset = {
      source: `Backend upload: ${data.filename}`,
      sourceType: "backend",
      columns: data.columns,
      preview: data.preview,
      rowCount: data.rows,
      columnCount: data.columns.length,
    };

    columns = data.columns;

    renderPreview(uploadedDataset);
    loadColumns();
    trainBtn.disabled = false;

    if (
      taskTypeSelect.value === "classification" ||
      taskTypeSelect.value === "regression"
    ) {
      targetDiv.classList.remove("is-hidden");
    }

    setStatus(
      `${data.filename} uploaded successfully. ${data.rows} rows, ${data.columns.length} columns detected.`,
      "success",
    );
  } catch (error) {
    uploadedDataset = null;
    columns = [];
    trainBtn.disabled = true;
    setEmptyState(
      previewPanel,
      "Upload a dataset to preview sample rows here.",
    );
    setStatus(error.message || "Unable to upload the dataset.", "error");
  } finally {
    uploadBtn.disabled = false;
    resetButtonLabel(uploadBtn);
  }
}

//train model
async function trainModel() {
  const task = taskTypeSelect.value;
  const target = document.getElementById("targetColumn").value;

  if (!uploadedDataset) {
    setStatus("Upload a dataset first.", "error");
    return;
  }

  if (!task) {
    setStatus("Select an ML task before starting training.", "error");
    return;
  }

  if ((task === "classification" || task === "regression") && !target) {
    setStatus(
      "Choose a target column for the selected supervised task.",
      "error",
    );
    return;
  }

  trainBtn.disabled = true;
  downloadBtn.disabled = true;
  setButtonLabel(trainBtn, "Training...");
  setStatus("Running AutoML pipeline on the backend...", "info");

  try {
    const body = { task_type: task };
    if (task === "classification" || task === "regression") {
      body.target_column = target;
    }
    // clustering: optimal k is now auto-detected by the backend

    const res = await fetch(`${BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Training failed.");
    }

    const data = await res.json();

    lastTrainingRun = {
      fileName: uploadedFile?.name || "dataset",
      task: task,
      target: target || null,
      trainedAt: new Date().toISOString(),
    };

    displayResults(data);
    downloadBtn.disabled = false;
    setStatus(`Training complete! Best model: ${data.best_model}`, "success");
  } catch (error) {
    setEmptyState(resultsPanel, "Run AutoML to display evaluation metrics.");
    setStatus(error.message || "Unable to complete training.", "error");
  } finally {
    trainBtn.disabled = false;
    resetButtonLabel(trainBtn);
  }
}

//download model
async function downloadModel() {
  if (!lastTrainingRun) {
    setStatus("Run AutoML training before downloading the model.", "error");
    return;
  }

  downloadBtn.disabled = true;
  setButtonLabel(downloadBtn, "Downloading...");
  setStatus("Downloading trained model from backend...", "info");

  try {
    const res = await fetch(`${BASE_URL}/export-model`);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Download failed.");
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "model.joblib";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    setStatus("Model downloaded successfully as model.joblib", "success");
  } catch (error) {
    setStatus(error.message || "Unable to download the model.", "error");
  } finally {
    downloadBtn.disabled = false;
    resetButtonLabel(downloadBtn);
  }
}
