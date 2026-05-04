/**
 * ComfyUI-HYWM2 — Splat Viewer extension.
 *
 * Hosts an iframe (viewer.html) inside the node and forwards the
 * .splat URL on execute. Persists viewer state (camera pose, opacity,
 * size, FOV, background, up-axis) into `node.properties` so saving the
 * workflow survives reloads — mirrors how GeometryPack persists
 * preview-camera state.
 */

import { app } from "/scripts/app.js";

const NODE_NAME = "HYWM2SplatAdvancedViewer";
const VIEWER_PATH = "/extensions/ComfyUI-HYWM2/splat_advanced/viewer.html";
const STATE_PROP = "hywm2_viewer_state";

app.registerExtension({
    name: "hywm2.splat_advanced_viewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // node.properties is auto-serialized into the workflow JSON
            // (LiteGraph's job), so writing here is enough — no DOM-widget
            // serialize plumbing needed.
            this.properties = this.properties || {};
            if (typeof this.properties[STATE_PROP] !== "string") {
                this.properties[STATE_PROP] = "{}";
            }

            const container = document.createElement("div");
            container.style.cssText = `
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                background: #1a1a1a;
                border-radius: 4px;
                overflow: hidden;
            `;

            const iframe = document.createElement("iframe");
            iframe.src = VIEWER_PATH;
            iframe.style.cssText = `
                width: 100%;
                height: 100%;
                border: none;
                background: #1a1a1a;
                display: block;
            `;
            iframe.allow = "fullscreen";
            container.appendChild(iframe);

            const widget = this.addDOMWidget(
                "splat_advanced_viewer",
                "HYWM2_SPLAT_ADVANCED_VIEWER",
                container,
                {
                    serialize: false,
                    hideOnZoom: false,
                    getValue: () => "",
                    setValue: () => {},
                }
            );
            widget.computeSize = () => [560, 520];

            this._hywm2_iframe = iframe;
            this._hywm2_pending = null;
            this._hywm2_pending_state = null;
            this._hywm2_ready = false;

            const node = this;
            const sendToIframe = (payload) => {
                if (node._hywm2_ready && node._hywm2_iframe?.contentWindow) {
                    node._hywm2_iframe.contentWindow.postMessage(payload, "*");
                    return true;
                }
                return false;
            };

            const onMessage = (event) => {
                if (!event?.data || event.source !== iframe.contentWindow) return;
                const d = event.data;

                if (d.type === "HYWM2_VIEWER_READY") {
                    node._hywm2_ready = true;
                    // 1. Restore saved viewer state if any.
                    let stateObj = null;
                    try { stateObj = JSON.parse(node.properties[STATE_PROP] || "{}"); }
                    catch { stateObj = {}; }
                    if (stateObj && Object.keys(stateObj).length) {
                        sendToIframe({ type: "HYWM2_RESTORE_STATE", state: stateObj });
                    }
                    // 2. If a state restore was queued before viewer was ready, flush.
                    if (node._hywm2_pending_state) {
                        sendToIframe(node._hywm2_pending_state);
                        node._hywm2_pending_state = null;
                    }
                    // 3. Same for any queued LOAD_SPLAT payload.
                    if (node._hywm2_pending) {
                        sendToIframe(node._hywm2_pending);
                        node._hywm2_pending = null;
                    }
                    return;
                }

                if (d.type === "HYWM2_VIEWER_STATE") {
                    // Iframe is reporting its current state — stash it on the node so
                    // the workflow JSON picks it up on save.
                    try {
                        node.properties[STATE_PROP] = JSON.stringify(d.state || {});
                    } catch (e) {
                        console.warn("[HYWM2 Splat Viewer] state serialize failed:", e);
                    }
                    return;
                }
            };
            window.addEventListener("message", onMessage);

            this.setSize([560, 540]);
            return r;
        };

        // onConfigure fires when LiteGraph restores a saved node from JSON.
        // `info.properties[STATE_PROP]` will contain whatever we stashed.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const node = this;
            const saved = info?.properties?.[STATE_PROP] ?? this.properties?.[STATE_PROP];
            if (!saved) return;
            let stateObj = null;
            try { stateObj = typeof saved === "string" ? JSON.parse(saved) : saved; }
            catch { stateObj = null; }
            if (!stateObj) return;
            // Cache for late delivery if iframe isn't ready yet.
            const payload = { type: "HYWM2_RESTORE_STATE", state: stateObj };
            if (node._hywm2_ready && node._hywm2_iframe?.contentWindow) {
                node._hywm2_iframe.contentWindow.postMessage(payload, "*");
            } else {
                node._hywm2_pending_state = payload;
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const ui = message?.ui ?? message ?? {};

            if (ui.error?.[0]) {
                console.warn("[HYWM2 Splat Viewer] error:", ui.error[0]);
                this._hywm2_iframe?.contentWindow?.postMessage(
                    { type: "HYWM2_LOAD_SPLAT", error: ui.error[0] },
                    "*"
                );
                return;
            }

            const url = ui.splat_url?.[0];
            const filename = ui.filename?.[0] || "(unknown)";
            const size = ui.file_size_bytes?.[0] || 0;
            const count = ui.gaussian_count?.[0] || 0;
            if (!url) return;

            const payload = {
                type: "HYWM2_LOAD_SPLAT",
                url, filename, size, count,
            };

            if (this._hywm2_ready && this._hywm2_iframe?.contentWindow) {
                this._hywm2_iframe.contentWindow.postMessage(payload, "*");
            } else {
                this._hywm2_pending = payload;
                if (this._hywm2_iframe) {
                    this._hywm2_iframe.addEventListener("load", () => {
                        setTimeout(() => {
                            if (!this._hywm2_ready && this._hywm2_pending) {
                                this._hywm2_iframe.contentWindow?.postMessage(
                                    this._hywm2_pending, "*"
                                );
                                this._hywm2_pending = null;
                            }
                        }, 500);
                    }, { once: true });
                }
            }
        };
    },
});
