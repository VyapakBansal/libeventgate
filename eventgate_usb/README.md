# Session 20260807_1136 sidecar

IMU CSV and timing for the paper recording (Lucid Triton2, Sony IMX637, 640x512).

**Not in git:** `events.h5` (~2.3 GB) and `events.raw` (~3.6 GB). Point `--events` at a local copy.

```
timestamp_us,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z
```

Times in `imu.csv` are already mapped into the event-clock domain (software co-stop, not PTP). IMU starts about 8.91 s after the first event. See `MANIFEST.json`.
