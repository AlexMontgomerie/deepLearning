# Get Dataset
if [ ! -d "hpatches" ]; then
  # TODO: wget hpatches
  wget -O hpatches_data.zip https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip
  unzip -q ./hpatches_data.zip
  rm ./hpatches_data.zip
fi

# get Benchmark
if [ ! -d "hpatches-benchmark" ]; then
  git clone https://github.com/hpatches/hpatches-benchmark
fi
