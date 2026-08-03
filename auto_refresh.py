import time
import pyautogui

interval_seconds = 60

print("Auto refresh started.")
print("Press Ctrl+C in terminal to stop.")

time.sleep(3)

while True:
    # macOS: switch to Chrome
    pyautogui.hotkey("command", "tab")
    time.sleep(0.5)

    # refresh
    pyautogui.hotkey("command", "r")
    print("Refreshed")
    time.sleep(interval_seconds)