docker build -t srp33/trial1 .
mkdir -p bioProjectIds
docker run --rm -i -t \
    -v $(pwd)/bioProjectIds:/bioProjectIds/ \
    -v $(pwd)/scripts:/scripts/ \
    -v $(pwd)/results:/results/ \
    -v $(pwd)/bioSamples:/bioSamples/ \
    srp33/trial1 \
    /exec_analysis.py