"""test_eyetracker.py
Test reading from Python eyetracker running in C++ using pipes.

Author: Ritwik Bera
"""
import time, sys
from subprocess import Popen, PIPE
import threading

class EyeTrackerThreaded(object):
    def __init__(self):
        # define gaze variables
        self.gaze_x = 0.
        self.gaze_y = 0.
        self.running_eyetracker = True

        # open pipe
        self.p = Popen(
            ['./start_eyetracker.sh'], shell=True, stdout=PIPE, stdin=PIPE)

        # update gaze values in separate thread
        self.gaze_thread = threading.Thread(target=self._update_gaze_values)
        self.gaze_thread.start()

    def _update_gaze_values(self):
        while self.running_eyetracker:
            # read data coming from cpp
            result_stdout = str(self.p.stdout.readline().strip()).split(',')

            # fix formatting and display
            self.gaze_x = float(result_stdout[0][1:])
            self.gaze_y = float(result_stdout[1][:-1])
            # print('inside class:', self.gaze_x, self.gaze_y)

    def close(self):
        # stops eye tracker thread
        self.running_eyetracker = False
        self.gaze_thread.join()

# create eye tracker object
eyetracker = EyeTrackerThreaded()

while True:
    try:
        # emulates delay caused by other computations
        time.sleep(0.5)

        # display eyetracker values
        print(eyetracker.gaze_x, eyetracker.gaze_y)

    except KeyboardInterrupt:
        # kill eyetracker thread when ctrl+C
        print('Killed with Ctrl+C')
        eyetracker.close()
        break

    


