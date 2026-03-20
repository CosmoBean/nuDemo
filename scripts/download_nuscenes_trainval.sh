#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROFILE="${NUSCENES_PROFILE:-keyframes}"
MODE="${NUSCENES_MODE:-all}"
KEEP_ARCHIVES="${NUSCENES_KEEP_ARCHIVES:-1}"
RAW_ROOT="${NUSCENES_RAW_ROOT:-${REPO_ROOT}/data/raw/v1.0-trainval}"
DATASET_ROOT="${NUSCENES_DATASET_ROOT:-${REPO_ROOT}/data/nuscenes}"
CHECKPOINT_ROOT="${REPO_ROOT}/artifacts/checkpoints/nuscenes-trainval/${PROFILE}"
LOG_ROOT="${REPO_ROOT}/artifacts/logs"
LOG_FILE="${LOG_ROOT}/nuscenes-trainval-${PROFILE}.log"
URL_BASE="https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0"

mkdir -p "${RAW_ROOT}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "missing required tool: $1"
        exit 1
    fi
}

file_size() {
    stat -c '%s' "$1"
}

download_marker() {
    local file="$1"
    printf '%s/%s.downloaded\n' "${CHECKPOINT_ROOT}" "${file}"
}

extract_marker() {
    local file="$1"
    printf '%s/%s.extracted\n' "${CHECKPOINT_ROOT}" "${file}"
}

verify_size() {
    local archive_path="$1"
    local expected_size="$2"
    local actual_size

    actual_size="$(file_size "${archive_path}")"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
        log "size mismatch for ${archive_path}: expected ${expected_size}, got ${actual_size}"
        return 1
    fi
    return 0
}

download_one() {
    local file="$1"
    local expected_size="$2"
    local archive_path="${RAW_ROOT}/${file}"
    local marker

    marker="$(download_marker "${file}")"
    if [[ -f "${marker}" && -f "${archive_path}" ]] && verify_size "${archive_path}" "${expected_size}"; then
        log "download checkpoint already satisfied for ${file}"
        return 0
    fi

    log "downloading ${file}"
    wget --continue --output-document="${archive_path}" "${URL_BASE}/${file}" >>"${LOG_FILE}" 2>&1
    verify_size "${archive_path}" "${expected_size}"
    printf '%s\n' "${expected_size}" >"${marker}"
    log "downloaded ${file}"
}

extract_one() {
    local file="$1"
    local archive_path="${RAW_ROOT}/${file}"
    local marker

    marker="$(extract_marker "${file}")"
    if [[ -f "${marker}" ]]; then
        log "extract checkpoint already satisfied for ${file}"
        return 0
    fi

    if [[ ! -f "${archive_path}" ]]; then
        log "cannot extract missing archive ${archive_path}"
        exit 1
    fi

    log "extracting ${file}"
    case "${file}" in
        *.tgz|*.tar.gz)
            tar -xzf "${archive_path}" -C "${DATASET_ROOT}"
            ;;
        *.zip)
            unzip -oq "${archive_path}" -d "${DATASET_ROOT}"
            ;;
        *)
            log "unsupported archive format: ${file}"
            exit 1
            ;;
    esac

    : >"${marker}"
    log "extracted ${file}"

    if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
        rm -f "${archive_path}"
        log "removed archive ${file}"
    fi
}

append_file() {
    FILES+=("$1")
    SIZES+=("$2")
}

declare -a FILES=()
declare -a SIZES=()

append_file "v1.0-trainval_meta.tgz" "461678030"
append_file "nuScenes-map-expansion-v1.3.zip" "398535531"
append_file "v1.0-trainval01_keyframes.tgz" "4529954279"
append_file "v1.0-trainval02_keyframes.tgz" "4272745444"
append_file "v1.0-trainval03_keyframes.tgz" "4249901872"
append_file "v1.0-trainval04_keyframes.tgz" "4562843980"
append_file "v1.0-trainval05_keyframes.tgz" "3957488692"
append_file "v1.0-trainval06_keyframes.tgz" "3856821591"
append_file "v1.0-trainval07_keyframes.tgz" "4177675764"
append_file "v1.0-trainval08_keyframes.tgz" "4282633755"
append_file "v1.0-trainval09_keyframes.tgz" "4814177310"
append_file "v1.0-trainval10_keyframes.tgz" "6198448085"

if [[ "${PROFILE}" == "full" ]]; then
    append_file "v1.0-trainval01_blobs.tgz" "31579122687"
    append_file "v1.0-trainval02_blobs.tgz" "30134721083"
    append_file "v1.0-trainval03_blobs.tgz" "29872679856"
    append_file "v1.0-trainval04_blobs.tgz" "32075538096"
    append_file "v1.0-trainval05_blobs.tgz" "28191611840"
    append_file "v1.0-trainval06_blobs.tgz" "27516468993"
    append_file "v1.0-trainval07_blobs.tgz" "29534216608"
    append_file "v1.0-trainval08_blobs.tgz" "30275496199"
    append_file "v1.0-trainval09_blobs.tgz" "33517622306"
    append_file "v1.0-trainval10_blobs.tgz" "41727447974"
fi

if [[ "${PROFILE}" != "keyframes" && "${PROFILE}" != "full" ]]; then
    log "unsupported profile: ${PROFILE}"
    exit 1
fi

if [[ "${MODE}" != "download" && "${MODE}" != "extract" && "${MODE}" != "all" ]]; then
    log "unsupported mode: ${MODE}"
    exit 1
fi

require_tool wget
require_tool tar
require_tool unzip

total_bytes=0
for size in "${SIZES[@]}"; do
    total_bytes="$((total_bytes + size))"
done
total_gib="$(awk "BEGIN {printf \"%.2f\", ${total_bytes}/1024/1024/1024}")"

log "profile=${PROFILE} mode=${MODE} keep_archives=${KEEP_ARCHIVES}"
log "raw_root=${RAW_ROOT}"
log "dataset_root=${DATASET_ROOT}"
log "checkpoint_root=${CHECKPOINT_ROOT}"
log "planned transfer=${total_gib} GiB across ${#FILES[@]} archives"

for idx in "${!FILES[@]}"; do
    file="${FILES[$idx]}"
    size="${SIZES[$idx]}"

    if [[ "${MODE}" == "download" || "${MODE}" == "all" ]]; then
        download_one "${file}" "${size}"
    fi

    if [[ "${MODE}" == "extract" || "${MODE}" == "all" ]]; then
        extract_one "${file}"
    fi
done

log "completed profile=${PROFILE} mode=${MODE}"
