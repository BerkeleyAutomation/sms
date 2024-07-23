from sms.tracking.zed import Zed
import time

tri_zed = Zed()
while True:
    start_time = time.time()
    left,right,depth = tri_zed.get_frame()
    end_time = time.time()
    print("Frequency " + str(1.0 / (end_time - start_time)) + " Hz")
print("HI")