#!/bin/sh

EXPECTED_SHA256="8db5cbddca227d8ee42132f1bd44f22d66c7840ed7dd0306454439c950be7a16"
MODEL_URL="https://store.aissquare.com/models/5e1232cd-c23f-489f-9db4-ee09592beeb1/DNN_model.zip"
ARCHIVE_NAME="DNN_model.zip"
MODEL_ENTRY="DNN_model/H2/DNN_model.pt"
MECH_ENTRY="DNN_model/H2/Burke2012_s9r23.yaml"

archive_has_expected_entries() {
    archive_path="$1"
    unzip -Z1 "$archive_path" | grep -qx "$MODEL_ENTRY" \
        && unzip -Z1 "$archive_path" | grep -qx "$MECH_ENTRY"
}

archive_matches_expected_hash() {
    archive_path="$1"
    if ! command -v sha256sum >/dev/null 2>&1; then
        echo "Warning: sha256sum not found; skipping SHA-256 verification for $archive_path" >&2
        return 0
    fi

    actual_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
    [ "$actual_sha256" = "$EXPECTED_SHA256" ]
}

validate_archive() {
    archive_path="$1"

    if [ ! -f "$archive_path" ]; then
        return 1
    fi

    if ! archive_has_expected_entries "$archive_path"; then
        echo "Archive $archive_path does not contain the expected H2 assets." >&2
        return 1
    fi

    if ! archive_matches_expected_hash "$archive_path"; then
        if command -v sha256sum >/dev/null 2>&1; then
            actual_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
            echo "Archive $archive_path has SHA-256 $actual_sha256, expected $EXPECTED_SHA256." >&2
        fi
        return 1
    fi

    return 0
}

if [ -e DNN_model.pt ] && [ -e Burke2012_s9r23.yaml ]; then
    echo "DNN_model.pt and Burke2012_s9r23.yaml already exist. Reusing local assets."
    exit 0
fi

archive_ok=false
if validate_archive "$ARCHIVE_NAME"; then
    archive_ok=true
    echo "Using local $ARCHIVE_NAME"
    if command -v sha256sum >/dev/null 2>&1; then
        echo "Local archive sha256: $(sha256sum "$ARCHIVE_NAME" | awk '{print $1}')"
    fi
else
    if [ -f "$ARCHIVE_NAME" ]; then
        echo "Local $ARCHIVE_NAME is invalid for the H2 backend examples; ignoring it."
    fi
fi

if [ "$archive_ok" != true ]; then
    for candidate in \
        ../pytorchEmbedded-H2/DNN_model.zip \
        ../onnxRuntime-H2/DNN_model.zip \
        ../tensorRt-H2/DNN_model.zip \
        ../../H2/DNN_model.zip
    do
        if validate_archive "$candidate"; then
            echo "Reusing DNN_model.zip from $candidate"
            cp "$candidate" "./$ARCHIVE_NAME" || exit 1
            archive_ok=true
            if command -v sha256sum >/dev/null 2>&1; then
                echo "Reused archive sha256: $(sha256sum "./$ARCHIVE_NAME" | awk '{print $1}')"
            fi
            break
        fi
    done
fi

if [ "$archive_ok" != true ]; then
    echo "DNN model assets not found locally with the expected SHA-256. Downloading $ARCHIVE_NAME ..."
    wget -O "$ARCHIVE_NAME" "$MODEL_URL" || exit 1
    if ! validate_archive "$ARCHIVE_NAME"; then
        echo "Downloaded $ARCHIVE_NAME failed validation; refusing to continue." >&2
        exit 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        echo "Downloaded archive sha256: $(sha256sum "$ARCHIVE_NAME" | awk '{print $1}')"
    fi
fi

unzip -o "$ARCHIVE_NAME" || exit 1
cp "./$MODEL_ENTRY" . || exit 1
cp "./$MECH_ENTRY" . || exit 1
