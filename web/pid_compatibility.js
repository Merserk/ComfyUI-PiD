export const PID_VERSIONS = Object.freeze(["v1", "v1.5"]);
export const PID_CKPT_TYPES = Object.freeze(["2k", "2kto4k"]);
export const PID_MODEL_PRECISIONS = Object.freeze(["bf16", "fp8", "int8"]);

export const PID_BACKBONES = Object.freeze([
    "zimage",
    "zimage-turbo",
    "flux",
    "flux2",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "sd3",
    "sdxl",
    "qwenimage",
    "qwenimage-2512",
]);

export const PID_UPSCALE_BACKBONES = Object.freeze([
    "zimage",
    "zimage-turbo",
    "flux",
    "flux2",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "sd3",
]);

const FLUX1_BACKBONES = new Set(["zimage", "zimage-turbo", "flux"]);
const FLUX2_BACKBONES = new Set(["flux2", "flux2-klein-4b", "flux2-klein-9b"]);
const QWEN_BACKBONES = new Set(["qwenimage", "qwenimage-2512"]);
const V15_BACKBONES = new Set([...FLUX1_BACKBONES, ...FLUX2_BACKBONES, ...QWEN_BACKBONES]);
const FOUR_K_ONLY_BACKBONES = new Set(["sdxl", ...QWEN_BACKBONES]);

function orderedSubset(source, allowed) {
    const allowedSet = new Set(allowed);
    return source.filter((value) => allowedSet.has(value));
}

function safeChoice(current, choices, preferred) {
    if (choices.includes(current)) {
        return current;
    }
    if (choices.includes(preferred)) {
        return preferred;
    }
    return choices[0];
}

export function pidBackboneChoices(version, allowedBackbones = PID_BACKBONES) {
    const allowed = orderedSubset(PID_BACKBONES, allowedBackbones);
    if (version === "v1.5") {
        return allowed.filter((backbone) => V15_BACKBONES.has(backbone));
    }
    return allowed;
}

export function pidCheckpointChoices(version, backbone) {
    if (version === "v1.5") {
        return V15_BACKBONES.has(backbone) ? ["2kto4k"] : [];
    }
    if (!PID_BACKBONES.includes(backbone)) {
        return [];
    }
    return FOUR_K_ONLY_BACKBONES.has(backbone) ? ["2kto4k"] : [...PID_CKPT_TYPES];
}

export function pidPrecisionChoices(version, backbone, pidCkptType) {
    if (!pidCheckpointChoices(version, backbone).includes(pidCkptType)) {
        return [];
    }
    if (version === "v1.5") {
        return ["bf16", "int8"];
    }
    if (FLUX1_BACKBONES.has(backbone)) {
        return ["bf16", "fp8"];
    }
    if (FLUX2_BACKBONES.has(backbone) && pidCkptType === "2k") {
        return ["bf16", "fp8"];
    }
    return ["bf16"];
}

export function normalizePidSelection(
    selection,
    { allowedBackbones = PID_BACKBONES, defaultBackbone = "zimage" } = {},
) {
    const version = PID_VERSIONS.includes(selection.version) ? selection.version : "v1";
    const backboneChoices = pidBackboneChoices(version, allowedBackbones);
    const backbone = safeChoice(selection.backbone, backboneChoices, defaultBackbone);
    const checkpointChoices = pidCheckpointChoices(version, backbone);
    const checkpointPreference = version === "v1.5" ? "2kto4k" : checkpointChoices[0];
    const pidCkptType = safeChoice(selection.pidCkptType, checkpointChoices, checkpointPreference);
    const precisionChoices = pidPrecisionChoices(version, backbone, pidCkptType);
    const modelPrecision = safeChoice(selection.modelPrecision, precisionChoices, "bf16");

    return {
        selection: { version, backbone, pidCkptType, modelPrecision },
        choices: {
            version: [...PID_VERSIONS],
            backbone: backboneChoices,
            pidCkptType: checkpointChoices,
            modelPrecision: precisionChoices,
        },
    };
}

export function emptyLatentBackboneChoices(pidCkptType, allowedBackbones = PID_BACKBONES) {
    const allowed = orderedSubset(PID_BACKBONES, allowedBackbones);
    if (pidCkptType === "2k") {
        return allowed.filter((backbone) => !FOUR_K_ONLY_BACKBONES.has(backbone));
    }
    return allowed;
}

export function normalizeEmptyLatentSelection(
    selection,
    { allowedBackbones = PID_BACKBONES, defaultBackbone = "sd3" } = {},
) {
    const pidCkptType = PID_CKPT_TYPES.includes(selection.pidCkptType)
        ? selection.pidCkptType
        : "2k";
    const backboneChoices = emptyLatentBackboneChoices(pidCkptType, allowedBackbones);
    const backbone = safeChoice(selection.backbone, backboneChoices, defaultBackbone);
    return {
        selection: { pidCkptType, backbone },
        choices: {
            pidCkptType: [...PID_CKPT_TYPES],
            backbone: backboneChoices,
        },
    };
}

export function listPidCombinations(allowedBackbones = PID_BACKBONES) {
    const combinations = [];
    for (const version of PID_VERSIONS) {
        for (const backbone of pidBackboneChoices(version, allowedBackbones)) {
            for (const pidCkptType of pidCheckpointChoices(version, backbone)) {
                for (const modelPrecision of pidPrecisionChoices(version, backbone, pidCkptType)) {
                    combinations.push({ version, backbone, pidCkptType, modelPrecision });
                }
            }
        }
    }
    return combinations;
}
