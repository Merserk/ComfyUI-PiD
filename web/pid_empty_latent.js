import { app } from "../../scripts/app.js";
import {
    PID_BACKBONES,
    PID_UPSCALE_BACKBONES,
    PID_VERSIONS,
    normalizeEmptyLatentSelection,
    normalizePidSelection,
} from "./pid_compatibility.js";

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

const PID_VERSION_CHOICES = new Set(PID_VERSIONS);
const PID_VERSION_WIDGET_INDEX = {
    PiDDecode: 0,
    PiDPrepare: 0,
    PiDUpscale: 1,
};

const PID_COMPATIBILITY_NODE_CONFIG = {
    PiDDecode: { allowedBackbones: PID_BACKBONES, defaultBackbone: "zimage" },
    PiDPrepare: { allowedBackbones: PID_BACKBONES, defaultBackbone: "zimage" },
    PiDUpscale: { allowedBackbones: PID_UPSCALE_BACKBONES, defaultBackbone: "flux" },
};

function arraysEqual(left, right) {
    return Array.isArray(left)
        && left.length === right.length
        && left.every((value, index) => value === right[index]);
}

function setWidgetChoices(widget, choices) {
    widget.options = widget.options ?? {};
    const current = widget.options.values;
    if (arraysEqual(current, choices)) {
        return false;
    }
    widget.options.values = [...choices];
    return true;
}

function setWidgetValue(widget, value) {
    if (widget.value === value) {
        return false;
    }
    widget.value = value;
    return true;
}

function wrapWidgetCallback(widget, marker, update) {
    if (!widget || widget[marker]) {
        return;
    }
    widget[marker] = true;
    const oldCallback = widget.callback;
    widget.callback = function pidCompatibilityCallback(value, ...args) {
        const result = oldCallback?.apply(this, [value, ...args]);
        update();
        return result;
    };
}

function installPiDCompatibilityFiltering(node, nodeName) {
    const config = PID_COMPATIBILITY_NODE_CONFIG[nodeName];
    if (!config || node.__pidCompatibilityFilteringInstalled) {
        return;
    }
    const widgets = Object.fromEntries((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const requiredNames = ["version", "backbone", "pid_ckpt_type", "model_precision"];
    if (requiredNames.some((name) => !widgets[name])) {
        return;
    }

    node.__pidCompatibilityFilteringInstalled = true;
    const update = () => {
        const normalized = normalizePidSelection(
            {
                version: widgets.version.value,
                backbone: widgets.backbone.value,
                pidCkptType: widgets.pid_ckpt_type.value,
                modelPrecision: widgets.model_precision.value,
            },
            config,
        );
        let changed = false;
        changed = setWidgetChoices(widgets.version, normalized.choices.version) || changed;
        changed = setWidgetChoices(widgets.backbone, normalized.choices.backbone) || changed;
        changed = setWidgetChoices(widgets.pid_ckpt_type, normalized.choices.pidCkptType) || changed;
        changed = setWidgetChoices(widgets.model_precision, normalized.choices.modelPrecision) || changed;
        changed = setWidgetValue(widgets.version, normalized.selection.version) || changed;
        changed = setWidgetValue(widgets.backbone, normalized.selection.backbone) || changed;
        changed = setWidgetValue(widgets.pid_ckpt_type, normalized.selection.pidCkptType) || changed;
        changed = setWidgetValue(widgets.model_precision, normalized.selection.modelPrecision) || changed;
        if (changed) {
            node.setDirtyCanvas(true, true);
        }
    };

    node.__pidUpdateCompatibility = update;
    wrapWidgetCallback(widgets.version, "__pidCompatibilityVersionCallback", update);
    wrapWidgetCallback(widgets.backbone, "__pidCompatibilityBackboneCallback", update);
    wrapWidgetCallback(widgets.pid_ckpt_type, "__pidCompatibilityCheckpointCallback", update);
    requestAnimationFrame(update);
}

function installEmptyLatentCompatibilityFiltering(node) {
    if (node.__pidEmptyLatentCompatibilityInstalled) {
        return;
    }
    const widgets = Object.fromEntries((node.widgets ?? []).map((widget) => [widget.name, widget]));
    if (!widgets.pid_ckpt_type || !widgets.backbone) {
        return;
    }

    node.__pidEmptyLatentCompatibilityInstalled = true;
    const update = () => {
        const normalized = normalizeEmptyLatentSelection({
            pidCkptType: widgets.pid_ckpt_type.value,
            backbone: widgets.backbone.value,
        });
        let changed = false;
        changed = setWidgetChoices(widgets.pid_ckpt_type, normalized.choices.pidCkptType) || changed;
        changed = setWidgetChoices(widgets.backbone, normalized.choices.backbone) || changed;
        changed = setWidgetValue(widgets.pid_ckpt_type, normalized.selection.pidCkptType) || changed;
        changed = setWidgetValue(widgets.backbone, normalized.selection.backbone) || changed;
        if (changed) {
            node.setDirtyCanvas(true, true);
        }
    };

    node.__pidUpdateCompatibility = update;
    wrapWidgetCallback(widgets.pid_ckpt_type, "__pidEmptyLatentCompatibilityCallback", update);
    requestAnimationFrame(update);
}

function installCompatibilityConfigureRefresh(nodeType) {
    if (nodeType.prototype.__pidCompatibilityConfigureRefreshInstalled) {
        return;
    }
    nodeType.prototype.__pidCompatibilityConfigureRefreshInstalled = true;
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function pidCompatibilityOnConfigure(info, ...args) {
        const result = onConfigure?.apply(this, [info, ...args]);
        requestAnimationFrame(() => this.__pidUpdateCompatibility?.());
        return result;
    };
}

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
    const requiredNames = ["auto_download", "upscale_factor", "strength"];
    if (requiredNames.some((name) => !widgets[name]) || node.__pidUpscaleDefaultResetInstalled) {
        return;
    }

    node.__pidUpscaleDefaultResetInstalled = true;

    const validUpscaleFactors = new Set(["2x", "4x", "6x", "8x"]);

    const resetInvalidValues = () => {
        let changed = false;
        if (typeof widgets.auto_download.value !== "boolean") {
            widgets.auto_download.value = true;
            changed = true;
        }
        if (!validUpscaleFactors.has(widgets.upscale_factor.value)) {
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

        const compatibilityConfig = PID_COMPATIBILITY_NODE_CONFIG[nodeData.name];
        if (compatibilityConfig || nodeData.name === "PiDEmptyLatentImage") {
            installCompatibilityConfigureRefresh(nodeType);
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function pidCompatibilityOnNodeCreated(...args) {
                const result = onNodeCreated?.apply(this, args);
                if (nodeData.name === "PiDEmptyLatentImage") {
                    installEmptyLatentCompatibilityFiltering(this);
                } else {
                    installPiDCompatibilityFiltering(this, nodeData.name);
                }
                return result;
            };
        }

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
