import threading

# abstract class
from abc import ABC, abstractmethod

from pynput import keyboard


class Controls(ABC):
    @abstractmethod
    def __init__(self, game_state):
        """Initialize the controls and start the listener thread"""
        pass

    @abstractmethod
    def listener(self):
        """Function run in the listener threads"""
        pass

    @abstractmethod
    def retrieve_keys(self):
        """Retrieve the pressed keys since last call"""
        pass

    @abstractmethod
    def get_action(self):
        """get the keys and translates it into actions (performable by the simulation )"""
        pass

    def quit(self):
        """Quit and cleans control handles threads ect"""
        pass

    def flush_keys(self):
        """Flush the keys"""
        pass


class KeyboardControlPynput:
    def __init__(self, game_state):
        self.game_state = game_state
        self.is_joystick = False

        # Keyboard keys
        self.CPG_keys = ["w", "s", "a", "d", "q"]

        # Shared lists to store key presses
        self.pressed_CPG_keys = []

        self.prev_gain_left = 0.0
        self.prev_gain_right = 0.0

        self._any_key_pressed = False

        self.lock = threading.Lock()

        print("Starting key listener")
        # Start the keyboard listener thread
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        key_str = (
            key.char if hasattr(key, "char") else str(key)
        )  # Gets the character of the key

        if key_str in self.CPG_keys:
            self.pressed_CPG_keys.append(key_str)
        if not self.game_state.get_reset():
            if key_str == "Key.space":
                self.game_state.set_reset(True)
        if key_str == "Key.esc":  # Quit when esc is pressed
            self.game_state.set_quit(True)
        with self.lock:
            self._any_key_pressed = True

    def any_key_pressed(self):
        with self.lock:
            return self._any_key_pressed

    def retrieve_keys(self):
        """Retrieve and clear all recorded key presses."""
        with self.lock:
            pCPG_keys = self.pressed_CPG_keys[:]
            self.pressed_CPG_keys.clear()

        return pCPG_keys

    def sort_keyboard_input(self, pCPG_keys):
        """Sorts the keys pressed and returns the last one."""
        keys = []
        if pCPG_keys:
            keys.append(max(set(pCPG_keys), key=pCPG_keys.count))

        return keys

    def get_actions(self):
        gain_left = self.prev_gain_left
        gain_right = self.prev_gain_right

        # Retrieve all keys pressed since the last call
        keys = self.sort_keyboard_input(self.retrieve_keys())

        for key in keys:
            if key == "a":
                if self.prev_gain_left < 0 or self.prev_gain_right < 0:
                    gain_right = -0.6
                    gain_left = -1.2  # Does not make sense but seems to generate the correct behavior (e.g outer leg has higher gain than outside leg ...)
                else:
                    gain_left = 0.4
                    gain_right = 1.2
            elif key == "d":
                if (
                    self.prev_gain_left < 0 or self.prev_gain_right < 0
                ):  # was going backwards previously
                    gain_left = -0.6  # same as above
                    gain_right = -1.2
                else:
                    gain_right = 0.4
                    gain_left = 1.2
            elif key == "w":
                gain_right = 1.0
                gain_left = 1.0
            elif key == "s":
                gain_right = -1.0
                gain_left = -1.0
            elif key == "q":
                gain_left = 0.0
                gain_right = 0.0

        self.prev_gain_left = gain_left
        self.prev_gain_right = gain_right

        return gain_left, gain_right

    def flush_keys(self):
        with self.lock:
            self.pressed_CPG_keys.clear()
            self._any_key_pressed = False

    def quit(self):
        self.listener.stop()
        self.listener.join()
        self.game_state.set_quit(True)
        self.flush_keys()
