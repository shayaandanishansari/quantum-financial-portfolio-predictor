import pyautogui
import time

INTERVAL = 60  # seconds between keypresses

print(f"Keep-awake running (pressing 'R' every {INTERVAL}s). Ctrl+C to stop.")

while True:
    time.sleep(INTERVAL)
    pyautogui.press('r')
    print(f"  [{time.strftime('%H:%M:%S')}] Pressed R")
