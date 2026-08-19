
import cv2
import numpy as np
import serial
 
# IMPORTANT: Update PORT to match your Arduino's actual port (e.g., 'COM3', 'COM5', or '/dev/cu.usbmodem...')
PORT = "/dev/cu.usbmodem101"
BAUD = 115200
WIDTH = 160
HEIGHT = 120
FRAME_SIZE = WIDTH * HEIGHT  # 1 byte per pixel for grayscale
 
SYNC1 = 0xAA
SYNC2 = 0x55
TYPE_IMAGE = 0x01
TYPE_STRING = 0x02
 
WINDOW_NAME = "OV7675 Photo (Grayscale)"
 
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Connected to {PORT} at {BAUD} baud.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()
 
# Clear out any stale bytes left in the OS buffer from before we opened the port.
ser.reset_input_buffer()
 
print("Waiting for messages from the board...")
print("With the image window focused, press 'q' to quit.")
 
 
def find_next_header():
    """Scan byte-by-byte for the 0xAA 0x55 sync header (so a dropped or stray
    byte can never leave us permanently misaligned), then read and return the
    message-type byte that follows it. Returns None if 'q' was pressed while
    waiting for data."""
    state = 0
    while True:
        b = ser.read(1)
        if not b:
            # No data yet -- keep the window responsive and check for quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None
            continue
        byte = b[0]
        if state == 0:
            if byte == SYNC1:
                state = 1
        else:
            if byte == SYNC2:
                type_byte = ser.read(1)
                if len(type_byte) == 1:
                    return type_byte[0]
                state = 0  # timed out reading the type byte -- keep scanning
            elif byte == SYNC1:
                state = 1  # this byte could itself be the start of a header
            else:
                state = 0
 
 
try:
    while True:
        msg_type = find_next_header()
        if msg_type is None:
            break
 
        if msg_type == TYPE_IMAGE:
            frame_data = ser.read(FRAME_SIZE)
            if len(frame_data) != FRAME_SIZE:
                print("Incomplete image frame received, discarding.")
                continue
 
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((HEIGHT, WIDTH))
            display = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(WINDOW_NAME, display)
            print("Photo captured and displayed.")
 
        elif msg_type == TYPE_STRING:
            length_byte = ser.read(1)
            if len(length_byte) != 1:
                print("Malformed string message, discarding.")
                continue
            str_len = length_byte[0]
            str_data = ser.read(str_len)
            if len(str_data) != str_len:
                print("Incomplete string message, discarding.")
                continue
            message = str_data.decode("utf-8", errors="replace")
            print(f"[Board] {message}")
 
        else:
            print(f"Unknown message type 0x{msg_type:02X}, skipping.")
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
except KeyboardInterrupt:
    print("Stopped.")
 
finally:
    ser.close()
    cv2.destroyAllWindows()
