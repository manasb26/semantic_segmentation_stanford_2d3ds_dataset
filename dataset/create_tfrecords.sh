# Script to preprocess the Stanford 2D3DS dataset.
#
# Usage:
#   bash ./create_tfrecords.sh
#
# The folder structure is assumed to be:
#  + dataset
#     - build_data.py
#     - build_sfd_2d3ds_data.py
#     - create_tfrecords.sh
#     + sfd_3d2ds_dataset
#         + dataset_devkits
#           + RGBImages
#           + SegmentedImages
#           + ImageSets    
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)

# Root path for the dataset.
SFD2D3DS_ROOT="${CURRENT_DIR}/sfd_3d2ds_dataset/dataset_devkit"
SEMANTIC_SEG_FOLDER="${SFD2D3DS_ROOT}/SegmentedImages"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${CURRENT_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${SFD2D3DS_ROOT}/RGBImages"
LIST_FOLDER="${SFD2D3DS_ROOT}/ImageSets"

echo "Creating TFRecords of Stanford 2D3DS dataset..."
python ./build_sfd_2d3ds_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"

echo "TFRecords of Stanford 2D3DS dataset are created!"  