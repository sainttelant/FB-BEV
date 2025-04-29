
import sys
import os
# append current dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_samples import *
from validate_trt_outputs import *