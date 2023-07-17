from deepspeed.launcher.runner import main
import os
import subprocess
if os.environ.get("path", None):
    if "/root/anaconda3/envs/env1/bin/" not in os.environ['PATH']:
        os.environ['PATH'] += ":/root/anaconda3/envs/env1/bin/"
else:
    os.environ['PATH'] = "/root/anaconda3/envs/env1/bin/"

if "/usr/bin/" not in os.environ['PATH']:
    os.environ['PATH'] += ":/usr/bin/"


if __name__ == '__main__':
    print(os.environ['PATH'])
    subprocess.check_output('ninja --version'.split())
    main([
        'deep_3.py',
        '--deepspeed_config=ds_config.json',
    ])

