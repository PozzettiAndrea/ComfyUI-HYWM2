// HYWM2 Preview + Filter Views -- DOM widget extension.
//
// Replaces the `enabled_mask` String widget on HYWM2PreviewAndFilterViews
// with a clickable thumbnail grid. Each thumbnail has an "enabled" overlay
// (transparent fill when ON, red X when OFF). Clicking toggles the view.
// The toggle state is JSON-encoded into the (hidden) enabled_mask widget
// value, so on the next prompt execution Python reads it back and emits
// only the enabled subset.

import { app } from "/scripts/app.js";

const NODE_ID = "HYWM2PreviewAndFilterViews";

function imageURL(view) {
  const params = new URLSearchParams({
    filename: view.filename,
    subfolder: view.subfolder,
    type: view.type || "temp",
  });
  return `/view?${params.toString()}&t=${Date.now()}`;
}

function fmtMatrix(m) {
  if (!m || !Array.isArray(m) || m.length === 0) return "(empty)";
  return m
    .map((row) =>
      row
        .map((v) => {
          const x = Number(v);
          if (!isFinite(x)) return "  nan ";
          return x.toFixed(3).padStart(7, " ");
        })
        .join(" "),
    )
    .join("\n");
}

function readMaskWidget(node) {
  // Returns the parsed boolean[] from the enabled_mask string widget, or null
  // if it's empty / not parseable. Caller falls back to all-true of length N.
  const w = node.widgets?.find((w) => w.name === "enabled_mask");
  if (!w) return null;
  const s = String(w.value || "").trim();
  if (!s) return null;
  try {
    const arr = JSON.parse(s);
    if (!Array.isArray(arr)) return null;
    return arr.map((v) => !!v);
  } catch {
    return null;
  }
}

function writeMaskWidget(node, mask) {
  const w = node.widgets?.find((w) => w.name === "enabled_mask");
  if (!w) return;
  w.value = JSON.stringify(mask);
  // Force-redraw so saving the workflow captures the new value.
  node.setDirtyCanvas?.(true, true);
}

function buildGrid(container, payload, node) {
  container.innerHTML = "";

  const { count, image_size, views } = payload;
  // Start from the saved widget mask if it matches N; else from the payload's
  // current `enabled_mask` (server state); else all true.
  let mask = readMaskWidget(node);
  if (!mask || mask.length !== count) {
    mask = Array.isArray(payload.enabled_mask) && payload.enabled_mask.length === count
      ? payload.enabled_mask.map((v) => !!v)
      : new Array(count).fill(true);
  }

  // Header summary + bulk-toggle buttons.
  const header = document.createElement("div");
  header.style.cssText =
    "display:flex;align-items:center;gap:8px;padding:6px 8px;font-size:11px;color:#bbb;border-bottom:1px solid #333;";
  const summary = document.createElement("span");
  const updateSummary = () => {
    const on = mask.filter(Boolean).length;
    summary.textContent = `${on}/${count} enabled  ·  ${image_size?.[0]}x${image_size?.[1]}`;
  };
  header.appendChild(summary);

  const spacer = document.createElement("span");
  spacer.style.flex = "1";
  header.appendChild(spacer);

  const mkBtn = (label, onClick) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText =
      "padding:2px 8px;background:#222;color:#ddd;border:1px solid #444;border-radius:3px;cursor:pointer;font-size:11px;";
    b.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
    return b;
  };
  header.appendChild(mkBtn("All", () => { mask = new Array(count).fill(true); commit(); }));
  header.appendChild(mkBtn("None", () => { mask = new Array(count).fill(false); commit(); }));
  header.appendChild(mkBtn("Invert", () => { mask = mask.map((v) => !v); commit(); }));
  container.appendChild(header);

  // Thumbnail grid.
  const grid = document.createElement("div");
  grid.style.cssText =
    "display:grid;grid-template-columns:repeat(auto-fill, minmax(140px, 1fr));gap:6px;padding:8px;overflow-y:auto;flex:1;";
  container.appendChild(grid);

  const cells = [];

  const updateCell = (i) => {
    const cell = cells[i];
    if (!cell) return;
    const on = mask[i];
    cell.style.borderColor = on ? "#5bbf6a" : "#bf4f4f";
    cell.style.opacity = on ? "1.0" : "0.4";
    cell.querySelector(".badge").textContent = on ? "ON" : "OFF";
    cell.querySelector(".badge").style.background = on ? "#1f5e29" : "#5e1f1f";
  };

  const commit = () => {
    writeMaskWidget(node, mask);
    updateSummary();
    for (let i = 0; i < count; i++) updateCell(i);
  };

  // Compute a sensible cell aspect ratio from the first view's image_size
  // so cells reserve space even before images load (or if URLs 404).
  const aspect = (image_size && image_size[1] && image_size[0])
    ? (image_size[1] / image_size[0]) : (9 / 16);

  views.forEach((v, i) => {
    const cell = document.createElement("div");
    cell.style.cssText =
      "position:relative;border:2px solid #5bbf6a;border-radius:4px;background:#222;overflow:hidden;cursor:pointer;user-select:none;min-height:80px;";

    // Reserve aspect-correct space so the cell stays the right shape even
    // if the <img> hasn't loaded yet or its URL is broken.
    const aspectBox = document.createElement("div");
    aspectBox.style.cssText = `position:relative;width:100%;padding-top:${(aspect * 100).toFixed(2)}%;`;
    cell.appendChild(aspectBox);

    const img = document.createElement("img");
    img.src = imageURL(v);
    img.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;display:block;";
    img.loading = "eager";  // 67 small thumbnails -- don't lazy-defer
    img.draggable = false;
    img.alt = `view ${v.index}`;

    // Visible placeholder + error surface (so a 404 becomes a "?" not a
    // collapsed-to-zero image).
    const errorOverlay = document.createElement("div");
    errorOverlay.style.cssText =
      "position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#bf4f4f;font-size:24px;font-weight:bold;background:#220;text-align:center;";
    errorOverlay.textContent = "?";
    img.addEventListener("error", () => {
      errorOverlay.style.display = "flex";
      errorOverlay.title = `image load failed:\n${img.src}`;
      console.error("[HYWM2PreviewAndFilterViews] image load failed", img.src);
    });

    aspectBox.appendChild(img);
    aspectBox.appendChild(errorOverlay);

    const badge = document.createElement("div");
    badge.className = "badge";
    badge.style.cssText =
      "position:absolute;top:4px;left:4px;padding:1px 6px;font-size:10px;font-weight:600;border-radius:3px;color:#fff;";
    cell.appendChild(badge);

    const idxLabel = document.createElement("div");
    idxLabel.textContent = `#${v.index}`;
    idxLabel.style.cssText =
      "position:absolute;bottom:4px;right:4px;padding:1px 6px;font-size:10px;background:rgba(0,0,0,0.6);color:#fff;border-radius:3px;font-family:ui-monospace,Menlo,Consolas,monospace;";
    cell.appendChild(idxLabel);

    cell.title =
      `View ${v.index}\n` +
      `extrinsics (w2c):\n${fmtMatrix(v.ext)}\n\n` +
      `intrinsics (K):\n${fmtMatrix(v.K)}`;

    cell.addEventListener("click", () => {
      mask[i] = !mask[i];
      commit();
    });

    grid.appendChild(cell);
    cells.push(cell);
  });

  // Initial paint
  for (let i = 0; i < count; i++) updateCell(i);
  updateSummary();
  // Persist the (possibly normalized) mask back to the widget so a fresh
  // save captures the canonical N-length array.
  writeMaskWidget(node, mask);
}

app.registerExtension({
  name: "HYWM2.PreviewAndFilterViews",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;

    const orig_onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig_onNodeCreated?.apply(this, arguments);

      // Hide the raw enabled_mask textbox -- the grid widget owns its value.
      // We keep the widget in the workflow (so the value serializes), but
      // shrink its computed size to ~0 so it disappears from the node UI.
      const maskWidget = this.widgets?.find((w) => w.name === "enabled_mask");
      if (maskWidget) {
        // Stash original computeSize for safety, then collapse to 0 height.
        maskWidget._hywm2_origComputeSize = maskWidget.computeSize;
        maskWidget.computeSize = () => [0, -4]; // negative height = effectively hidden
        // Also nuke draw + input handling so it never accepts clicks.
        maskWidget.draw = () => {};
        maskWidget.onMouseDown = () => false;
      }

      const container = document.createElement("div");
      container.style.cssText = `
        background:#111;
        border:1px solid #333;
        border-radius:4px;
        font-family:ui-sans-serif,system-ui,sans-serif;
        color:#ddd;
        display:flex;
        flex-direction:column;
        min-height:160px;
        overflow:hidden;
      `;
      const placeholder = document.createElement("div");
      placeholder.style.cssText =
        "padding:18px;text-align:center;color:#666;font-style:italic;";
      placeholder.textContent =
        "(queue the workflow to populate -- all views ON by default)";
      container.appendChild(placeholder);

      const widget = this.addDOMWidget("preview_views_grid", "div", container, {
        serialize: false,
        hideOnZoom: false,
      });
      widget.computeSize = () => [this.size[0], 420];

      this._hywm2PreviewContainer = container;

      if (!this.size || this.size[0] < 620) this.size = [720, 540];
    };

    const orig_onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      orig_onExecuted?.apply(this, arguments);
      const payloads = message?.preview_views;
      if (!payloads || payloads.length === 0) return;
      let payload;
      try {
        payload = JSON.parse(payloads[0]);
      } catch (err) {
        console.error("[HYWM2PreviewAndFilterViews] bad payload", err);
        return;
      }
      // Diagnostic: log the first image URL so the user can paste it into
      // a new browser tab to see exactly what the /view endpoint returns
      // (200 image, 404 with text, 403, etc.).
      if (payload.views && payload.views.length > 0) {
        const v0 = payload.views[0];
        const url0 =
          `/view?filename=${encodeURIComponent(v0.filename)}` +
          `&subfolder=${encodeURIComponent(v0.subfolder)}` +
          `&type=${encodeURIComponent(v0.type || "temp")}`;
        console.log(
          `[HYWM2PreviewAndFilterViews] payload: count=${payload.count} ` +
          `subfolder=${v0.subfolder} first-url=${url0}`,
        );
      }
      if (this._hywm2PreviewContainer) {
        buildGrid(this._hywm2PreviewContainer, payload, this);
      }
    };
  },
});
