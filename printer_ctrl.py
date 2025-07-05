import time, sys, csv, itertools, numpy as np, serial

PORT = "/dev/cu.usbserial-2120"  # or /dev/ttyACM0 depending on Pi
BAUD = 115200           # default for Creality boards
FEED = 3000             # mm · min⁻¹ – safe, quiet

# ---------- serial helpers ----------
def open_printer(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)
    # optional: silence temp autotalk so it doesn't clutter reads
    ser.write(b"M155 S0\n")       # disable auto-temperature reports
    wait_ok(ser)                  # now expect the first "ok"
    return ser

def send(ser, cmd):
    ser.write((cmd + "\n").encode())
    wait_ok(ser)

def wait_ok(ser):
    while True:
        line = ser.readline().decode().strip()
        if line == "ok":
            break

# ---------- motion ----------
def move_hydrophone(ser, x, y, z):
    send(ser, f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{FEED}")
    send(ser, "M400")        # block until motion complete

# ---------- ultrasound / hydrophone ----------
def ultrasound_pulse(device):
    device.fire()            # your driver abstraction

def record_hydrophone(hydrophone):
    return hydrophone.measure_ispta()  # returns scalar

# ---------- sweep generator ----------
def cartesian_grid(origin=(0,0,0), size=50, step=1):
    ox, oy, oz = origin
    rng = range(0, size, step)
    # serpentine pattern to minimise long travel moves in X
    for z in rng:
        for y in rng:
            xs = rng if (y+z) % 2 == 0 else reversed(rng)
            for x in xs:
                yield ox+x, oy+y, oz+z

# ---------- calibration routine --------
import sys, termios, tty

# Move step per keypress in mm
JOG_STEP = 3     # e.g., half millimeter per keypress

def get_key():
    """Read single keypress without Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def calibrate(ser):
    """Manual jog calibration routine."""
    x, y, z = 0.0, 0.0, 0.0
    print("\nCalibration Mode:")
    print("A/D = -X/+X | W/S = +Y/-Y | Q/E = +Z/-Z")
    print("Press [Enter] when positioned correctly.")

    while True:
        key = get_key()
        if key == '\r' or key == '\n':   # Enter pressed
            print(f"Calibrated at X={x:.2f} Y={y:.2f} Z={z:.2f}")
            break
        elif key.lower() == 'a':
            x -= JOG_STEP
        elif key.lower() == 'd':
            x += JOG_STEP
        elif key.lower() == 'w':
            y += JOG_STEP
        elif key.lower() == 's':
            y -= JOG_STEP
        elif key.lower() == 'q':
            z += JOG_STEP
        elif key.lower() == 'e':
            z -= JOG_STEP
        else:
            print(f"Ignored key: {repr(key)}")
            continue

        move_hydrophone(ser, x, y, z)

    return x, y, z

# ---------- main ----------
def main():
    #pulse_cfg   = {}  # read from CSV / args
    #hydro_cfg   = {}
    #device      = Device(pulse_cfg)
    #hydrophone  = Hydrophone(hydro_cfg)

    ser = open_printer()

    x,y,z = calibrate(ser)
    print(x,y,z)
    exit(-1)

    voxels = (50, 50, 50)        # 50 mm cube, 1 mm step
    data   = np.zeros(voxels)

    for idx, (x, y, z) in enumerate(cartesian_grid(step=1)):
        move_hydrophone(ser, x, y, z)
        ultrasound_pulse(device)               # excite
        data[x, y, z] = record_hydrophone(hydrophone)
        if idx % 1000 == 0:
            print(f"{idx/125000:.1%} done")

    np.save("hydro_scan.npy", data)
    send(ser, "M84")   # disable steppers
    ser.close()

if __name__ == "__main__":
    main()

