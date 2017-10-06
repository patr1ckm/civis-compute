# You can run this example via
#
#    $ civis-compute submit zombie.py
#    $ <JOBID>
#    $ civis-compute status
#    $ civis-compute cancel <JOBID>
#
# Make sure to cancel the job or it will run for a long time.

import time

#CIVIS name=zombie  # the name of the job in Civis Platform
#CIVIS required_resources={cpu: 256, memory: 1024, disk_space: 1}
#CIVIS docker_image_name=civisanalytics/datascience-python
#CIVIS docker_image_tag=3.2.0

t0 = time.time()
while True:
    if time.time() - t0 > 10:
        print('oooooooooooh!', flush=True)
        t0 = time.time()
