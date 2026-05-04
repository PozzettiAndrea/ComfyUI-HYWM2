/**
 * ComfyUI-HYWM2 — PLY Advanced Gaussian Viewer extension.
 *
 * Hosts an iframe (viewer.html) inside the node, forwarding the PLY URL
 * computed server-side after each execution. The viewer itself is a fully
 * standalone HTML page — see ./viewer.html.
 */

import { app } from "/scripts/app.js";

const NODE_NAME = "HYWM2PLYAdvancedGaussianViewer";
const VIEWER_PATH = "/extensions/ComfyUI-HYWM2/ply_advanced_gaussian/viewer.html";

app.registerExtension({
    name: "hywm2.ply_advanced_gaussian_viewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

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
                "ply_advanced_viewer",
                "HYWM2_PLY_ADVANCED_VIEWER",
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
            this._hywm2_ready = false;

            const onMessage = (event) => {
                if (!event?.data) return;
                if (event.data.type === "HYWM2_VIEWER_READY") {
                    this._hywm2_ready = true;
                    if (this._hywm2_pending) {
                        iframe.contentWindow?.postMessage(this._hywm2_pending, "*");
                        this._hywm2_pending = null;
                    }
                }
            };
            window.addEventListener("message", onMessage);

            this.setSize([560, 540]);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const ui = message?.ui ?? message ?? {};

            if (ui.error?.[0]) {
                console.warn("[HYWM2 PLY Viewer] error:", ui.error[0]);
                this._hywm2_iframe?.contentWindow?.postMessage(
                    { type: "HYWM2_LOAD_PLY", error: ui.error[0] },
                    "*"
                );
                return;
            }

            const url = ui.ply_url?.[0];
            const filename = ui.filename?.[0] || "(unknown)";
            const size = ui.file_size_bytes?.[0] || 0;
            if (!url) return;

            const payload = {
                type: "HYWM2_LOAD_PLY",
                url,
                filename,
                size,
            };

            if (this._hywm2_ready && this._hywm2_iframe?.contentWindow) {
                this._hywm2_iframe.contentWindow.postMessage(payload, "*");
            } else {
                this._hywm2_pending = payload;
                if (this._hywm2_iframe) {
                    this._hywm2_iframe.addEventListener(
                        "load",
                        () => {
                            // viewer should post HYWM2_VIEWER_READY soon after; fall back after 500ms.
                            setTimeout(() => {
                                if (!this._hywm2_ready && this._hywm2_pending) {
                                    this._hywm2_iframe.contentWindow?.postMessage(
                                        this._hywm2_pending,
                                        "*"
                                    );
                                    this._hywm2_pending = null;
                                }
                            }, 500);
                        },
                        { once: true }
                    );
                }
            }
        };
    },
});
