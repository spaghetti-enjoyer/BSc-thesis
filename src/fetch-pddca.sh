#!/bin/bash
set -e

OUTPUT_DIR="parotid_PDDCA"
BASE_URL="https://www.imagenglab.com/data/pddca"

echo "=== Downloading PDDCA 1.4.1 ==="

# download all 3 parts
for i in 1 2 3; do
    FILE="PDDCA-1.4.1_part${i}.zip"
    if [ -f "$FILE" ]; then
        echo "Part $i already downloaded, skipping."
    else
        echo "Downloading part $i..."
        wget -q --show-progress "${BASE_URL}/${FILE}"
    fi
done

echo ""
echo "=== Unzipping ==="

mkdir -p "$OUTPUT_DIR"

for i in 1 2 3; do
    echo "Unzipping part $i..."
    unzip -q "PDDCA-1.4.1_part${i}.zip" -d "part${i}_tmp"
done

echo ""
echo "=== Merging into ${OUTPUT_DIR}/ ==="

# each zip extracts a folder — move all patient subdirs into the single output dir
for i in 1 2 3; do
    # find the top-level extracted folder (whatever it's named)
    EXTRACTED=$(find "part${i}_tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
    echo "Part $i extracted to: $EXTRACTED"
    # move each patient folder into the output dir
    for patient_dir in "$EXTRACTED"/*/; do
        pid=$(basename "$patient_dir")
        if [ -d "${OUTPUT_DIR}/${pid}" ]; then
            echo "  WARNING: ${pid} already exists, skipping duplicate."
        else
            mv "$patient_dir" "${OUTPUT_DIR}/"
        fi
    done
done

echo ""
echo "=== Cleaning up ==="
rm -rf part1_tmp part2_tmp part3_tmp
rm -f PDDCA-1.4.1_part1.zip PDDCA-1.4.1_part2.zip PDDCA-1.4.1_part3.zip

echo ""
echo "=== Done ==="
echo "Patients found: $(ls ${OUTPUT_DIR} | wc -l)"
echo "Location: ${OUTPUT_DIR}/"
ls "$OUTPUT_DIR"