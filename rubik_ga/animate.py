import cv2
import imageio
import numpy as np
import tkinter as tk
from typing import List, Tuple

def create_frame(size: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    return np.full((size[0], size[1], 3), color, dtype=np.uint8)

def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
    cv2.circle(frame, center, radius, color, -1)

def export_to_gif(frames: List[np.ndarray], filename: str, duration: float) -> None:
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    imageio.mimsave(filename, rgb_frames, duration=duration)

def play_animation(duration: float) -> None:
    frames = []
    for i in range(51):
        frame = create_frame((300, 300), (255, 255, 255))
        draw_circle(frame, (i * 5, i * 5), 20, (0, 0, 255))
        frames.append(frame)
        cv2.imshow('Animation', frame)
        cv2.waitKey(int(1000 * duration))

    cv2.destroyAllWindows()
    export_to_gif(frames, 'my_animation.gif', duration)

def main():
    root = tk.Tk()
    root.title("Animate")

    play_button = tk.Button(root, text="Play", command=lambda: play_animation(0.05))
    play_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()