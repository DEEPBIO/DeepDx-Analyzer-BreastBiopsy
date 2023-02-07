# DeepDx-Analyzer
Package for inference on whole slide images(WSIs) of Breast Resection.

## Usage

### **To use in other projects:** 
1. Pip install this repository or add this repo as submodule
```bash
# Direct install via pip. 
# Note that you might need to set up ssh key. 
# For more information, visit https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh
pip install git+ssh://git@github.com/DEEPBIO/DeepDx-Analyzer-Breast

# Adding as submodule
git submodule add https://github.com/DEEPBIO/DeepDx-Analyzer-Breast
cd DeepDx-Analyzer-Breast
pip install .
```

2. Install appropriate dependencies (see requirements.txt)

3. write code & run
```python
from DeepDxAnalyzerBreast import BreastResectionAnalyzer

slide = '/path/to/slide'
prostateAnalyzer = ProstateAnalyzer() # Creating instance of ProstateAnalyzer loads model to GPU
result = prostateAnalyzer.analyze(slide, [(0,0), (5000, 5000)])
```

### **For developing in this repository**
1. git clone this repo
```bash
git clone https://github.com/DEEPBIO/DeepDx-Analyzer-Breast
```
2. Build docker images
```bash
docker build --tag image-name .
```

3. Run in docker env
```bash
docker run -it -v $(pwd):/root/code image-name bash
```

## Testing
1. build image for testing
```bash
docker build --tag deepdx-analyzer-unittest .
```

2. Run test
```bash
./unittest.sh
```
for more information about testing, see README.md in `./tests`
