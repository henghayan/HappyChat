import subprocess
import os
import sys


if '/usr/local/cuda-11.7/bin' not in os.environ['PATH']:
    os.environ['PATH'] += ':/usr/local/cuda-11.7/bin/'

print(subprocess.check_output('nvcc --version'.split()))
