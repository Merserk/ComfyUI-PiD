import { app } from "../../scripts/app.js";

const RESOLUTION_CHOICES = {
    "2k": [
        "512x512 (1:1)",
        "576x432 (4:3)",
        "432x576 (3:4)",
        "624x416 (3:2)",
        "416x624 (2:3)",
        "672x384 (16:9)",
        "384x672 (9:16)",
        "784x336 (21:9)",
        "336x784 (9:21)",
    ],
    "2kto4k": [
        "1024x1024 (1:1)",
        "1024x768 (4:3)",
        "768x1024 (3:4)",
        "1008x672 (3:2)",
        "672x1008 (2:3)",
        "1024x576 (16:9)",
        "576x1024 (9:16)",
        "1008x432 (21:9)",
        "432x1008 (9:21)",
    ],
};

const PID_VERSION_CHOICES = new Set(["v1", "v1.5"]);
const PID_VERSION_WIDGET_INDEX = {
    PiDDecode: 0,
    PiDPrepare: 0,
    PiDUpscale: 1,
};

function installPiDVersionMigration(nodeType, nodeName) {
    const versionIndex = PID_VERSION_WIDGET_INDEX[nodeName];
    if (versionIndex === undefined || nodeType.prototype.__pidVersionMigrationInstalled) {
        return;
    }

    nodeType.prototype.__pidVersionMigrationInstalled = true;
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function pidVersionOnConfigure(info, ...args) {
        const values = info?.widgets_values;
        if (Array.isArray(values) && !PID_VERSION_CHOICES.has(values[versionIndex])) {
            values.splice(versionIndex, 0, "v1");
        }
        return onConfigure?.apply(this, [info, ...args]);
    };
}

function ratioFromLabel(label) {
    return String(label ?? "").match(/\(([^)]+)\)/)?.[1] ?? "1:1";
}

function installResolutionSwitcher(node) {
    const modeWidget = node.widgets?.find((widget) => widget.name === "pid_ckpt_type");
    const resolutionWidget = node.widgets?.find((widget) => widget.name === "resolution");
    if (!modeWidget || !resolutionWidget || resolutionWidget.__pidResolutionSwitcherInstalled) {
        return;
    }

    resolutionWidget.__pidResolutionSwitcherInstalled = true;

    const updateResolutionChoices = () => {
        const mode = modeWidget.value === "2kto4k" ? "2kto4k" : "2k";
        const choices = RESOLUTION_CHOICES[mode];
        const previousRatio = ratioFromLabel(resolutionWidget.value);

        if (resolutionWidget.options) {
            resolutionWidget.options.values = choices;
        }

        if (!choices.includes(resolutionWidget.value)) {
            resolutionWidget.value = choices.find((choice) => ratioFromLabel(choice) === previousRatio) ?? choices[0];
        }

        node.setDirtyCanvas(true, true);
    };

    const oldModeCallback = modeWidget.callback;
    modeWidget.callback = function pidModeCallback(value, ...args) {
        const result = oldModeCallback?.apply(this, [value, ...args]);
        updateResolutionChoices();
        return result;
    };

    // Run once after Comfy has restored widget values from workflow JSON.
    requestAnimationFrame(updateResolutionChoices);
}

function installUpscaleStrengthReset(node) {
    const strengthWidget = node.widgets?.find((widget) => widget.name === "strength");
    if (!strengthWidget || strengthWidget.__pidStrengthResetInstalled) {
        return;
    }

    strengthWidget.__pidStrengthResetInstalled = true;

    const resetStrengthToNumber = () => {
        const value = Number(strengthWidget.value);
        if (!Number.isFinite(value)) {
            strengthWidget.value = 0.4;
        } else {
            strengthWidget.value = Math.min(1, Math.max(0, Math.round(value * 10) / 10));
        }
        node.setDirtyCanvas(true, true);
    };

    requestAnimationFrame(resetStrengthToNumber);
}

function installUpscaleDefaultReset(node) {
    const widgets = Object.fromEntries((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const requiredNames = ["pid_ckpt_type", "version", "backbone", "auto_download", "model_precision", "upscale_factor", "strength"];
    if (requiredNames.some((name) => !widgets[name]) || node.__pidUpscaleDefaultResetInstalled) {
        return;
    }

    node.__pidUpscaleDefaultResetInstalled = true;

    const valid = {
        pid_ckpt_type: new Set(["2k", "2kto4k"]),
        version: PID_VERSION_CHOICES,
        backbone: new Set(["zimage", "zimage-turbo", "flux", "flux2", "flux2-klein-4b", "flux2-klein-9b", "sd3"]),
        model_precision: new Set(["bf16", "fp8", "int8"]),
        upscale_factor: new Set(["2x", "4x", "6x", "8x"]),
    };

    const resetInvalidValues = () => {
        let changed = false;
        if (!valid.pid_ckpt_type.has(widgets.pid_ckpt_type.value)) {
            widgets.pid_ckpt_type.value = "2k";
            changed = true;
        }
        if (!valid.version.has(widgets.version.value)) {
            widgets.version.value = "v1";
            changed = true;
        }
        if (!valid.backbone.has(widgets.backbone.value)) {
            widgets.backbone.value = "flux";
            changed = true;
        }
        if (typeof widgets.auto_download.value !== "boolean") {
            widgets.auto_download.value = true;
            changed = true;
        }
        if (!valid.model_precision.has(widgets.model_precision.value)) {
            widgets.model_precision.value = "bf16";
            changed = true;
        }
        if (!valid.upscale_factor.has(widgets.upscale_factor.value)) {
            widgets.upscale_factor.value = "4x";
            changed = true;
        }
        const strength = Number(widgets.strength.value);
        if (!Number.isFinite(strength) || strength < 0 || strength > 1) {
            widgets.strength.value = 0.4;
            changed = true;
        }
        if (changed) {
            node.setDirtyCanvas(true, true);
        }
    };

    requestAnimationFrame(resetInvalidValues);
}

function installCaptionPreview(node) {
    const previewWidget = node.widgets?.find((widget) => widget.name === "preview");
    if (!previewWidget || previewWidget.__pidCaptionPreviewInstalled) {
        return;
    }

    previewWidget.__pidCaptionPreviewInstalled = true;
    previewWidget.inputEl?.setAttribute?.("readonly", "readonly");
    previewWidget.inputEl?.setAttribute?.("disabled", "disabled");
    previewWidget.options = previewWidget.options ?? {};
    previewWidget.options.readonly = true;
}

app.registerExtension({
    name: "ComfyUI-PiD.Widgets",
    beforeRegisterNodeDef(nodeType, nodeData) {
        installPiDVersionMigration(nodeType, nodeData.name);

        if (nodeData.name === "PiDEmptyLatentImage") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function pidEmptyLatentOnNodeCreated(...args) {
                const result = onNodeCreated?.apply(this, args);
                installResolutionSwitcher(this);
                return result;
            };
        }

        if (nodeData.name === "PiDUpscale") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function pidUpscaleOnNodeCreated(...args) {
                const result = onNodeCreated?.apply(this, args);
                installUpscaleDefaultReset(this);
                installUpscaleStrengthReset(this);
                return result;
            };
        }

        if (nodeData.name === "PiDCaptionCreator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function pidCaptionCreatorOnNodeCreated(...args) {
                const result = onNodeCreated?.apply(this, args);
                installCaptionPreview(this);
                return result;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function pidCaptionCreatorOnExecuted(message, ...args) {
                const result = onExecuted?.apply(this, [message, ...args]);
                const previewWidget = this.widgets?.find((widget) => widget.name === "preview");
                const text = message?.text?.[0] ?? message?.caption?.[0];
                if (previewWidget && text !== undefined) {
                    previewWidget.value = String(text);
                    this.setDirtyCanvas(true, true);
                }
                return result;
            };
        }
    },
});
