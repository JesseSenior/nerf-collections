#!/bin/bash
cd "$(dirname "$0")/.."

rm -r data
mkdir -p data

echo "Downloading example data:"
wget -q --show-progress -c "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
fi

cd data
echo -n "Unzipping example data:"
unzip ../nerf_example_data.zip | awk 'BEGIN {ORS=" "} {if(NR%25==0)print "."}'
if [ $? -ne 0 ]; then
    echo "Unzip failed! Please check if 'unzip'is installed!"
    exit 1
fi

echo.

for folder in */; do
    mv "$folder"* ./
    rmdir "$folder"
done

rm ../nerf_example_data.zip 
echo "Done."