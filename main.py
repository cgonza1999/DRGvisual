import tkinter as tk
from tkinter import font as tkfont  # Import tkfont for custom font

import cv2
import numpy as np

import cv_functions as cvf
import display_functions as df


class ImageLabelerApp:
    # Initialization constructor
    def __init__(self, root):
        # Construct main UI
        self.root = root  # Main root program window
        self.root.state('zoomed')  # Maximize the main application window
        self.canvas = tk.Canvas(self.root, bg='gray')  # Create a canvas with a gray background
        self.canvas.pack(fill='both', expand=True)  # Make the canvas fill the window

        # Construct Image Arrays
        self.cv2_images = []  # Array to store raw cv2 image data
        self.photoimages = []  # Array to store PhotoImage objects for display
        self.image_file_paths = []  # Stores the file paths of the images
        self.image_labels = []  # Stores the labels for the images
        self.DRG_line_coords = []
        self.DRG_radii = []
        self.DRG_segments = []

        # Construct default indices and display values
        self.image_label_display = None  # Store the label widget for reference
        self.current_image_index = None  # Tracks the current image

        # Add dimensions for default image window
        self.rect_start_x = int(0.1 * self.root.winfo_screenwidth())
        self.rect_start_y = int(0.1 * self.root.winfo_screenheight())
        self.rect_end_x = int(0.9 * self.root.winfo_screenwidth())
        self.rect_end_y = int(0.9 * self.root.winfo_screenheight())
        self.rect_width = self.rect_end_x - self.rect_start_x
        self.rect_height = self.rect_end_y - self.rect_start_y
        self.rect_center_x = int(self.rect_start_x + self.rect_width / 2)
        self.rect_center_y = int(self.rect_start_y + self.rect_height / 2)

        # Create buttons
        self.prev_button = tk.Button(self.root, text="< Prev", command=lambda: df.prev_image(self))
        self.next_button = tk.Button(self.root, text="Next >", command=lambda: df.next_image(self))
        self.select_button = tk.Button(self.root, text="Select Image Files",
                                       command=lambda: cvf.select_and_load_files(self))
        self.rgb_split_button = tk.Button(self.root, text="RGB Preview", command=lambda: df.rgb_split(self))
        self.red_channel_button = tk.Button(self.root, text="R", command=lambda: cvf.isolate_channel(self, "r"))
        self.green_channel_button = tk.Button(self.root, text="G", command=lambda: cvf.isolate_channel(self, "g"))
        self.blue_channel_button = tk.Button(self.root, text="B", command=lambda: cvf.isolate_channel(self, "b"))
        self.rgb_restore_button = tk.Button(self.root, text="Restore", command=lambda: cvf.rgb_restore(self))
        self.drg_segment_button = tk.Button(self.root, text="DRG Segmentation", command=lambda: cvf.drg_segment(self))

        # Track mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel_scroll)

        # Initialize variables
        self.pan_start_x = None
        self.pan_start_y = None
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.zoom_level = 1.0
        self.draw_start_x = None
        self.draw_start_y = None
        self.draw_end_x = None
        self.draw_end_y = None

        # Call UI setup
        self.root.after(100, self.setup_ui())

    def on_mouse_press(self, event):
        # Start panning
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_mouse_drag(self, event):
        # Pan the image
        if self.pan_start_x is not None and self.pan_start_y is not None:
            delta_x = event.x - self.pan_start_x
            delta_y = event.y - self.pan_start_y
            self.image_offset_x += delta_x * self.zoom_level
            self.image_offset_y += delta_y * self.zoom_level
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            df.display_current_image(self)

    def on_mouse_wheel_scroll(self, event):
        # Zoom in or out based on mouse wheel scroll
        image = self.cv2_images[self.current_image_index]

        zoomed_image = np.ndarray(shape=(1, 1))

        if event.delta > 0:
            self.zoom_level * 1.1
            zoomed_image = cv2.resize(image, (int(image.shape[0] * 1.1), int(image.shape[1] * 1.1)),
                                      interpolation=cv2.INTER_CUBIC)
        else:
            self.zoom_level / 1.1
            zoomed_image = cv2.resize(image, (int(image.shape[0] / 1.1), int(image.shape[1] / 1.1)),
                                      interpolation=cv2.INTER_CUBIC)

        self.cv2_images[self.current_image_index] = zoomed_image
        df.display_current_image(self)

    def setup_ui(self):
        # Setup default image window
        self.canvas.create_rectangle(self.rect_start_x, self.rect_start_y,
                                     self.rect_end_x, self.rect_end_y,
                                     fill="black", outline="black")

        # Calculate center of default image window for label position
        rect_center_x = self.rect_start_x + (self.rect_end_x - self.rect_start_x) / 2

        # Create a label widget for displaying image label
        label_font = tkfont.Font(family="Arial", size=14, weight="bold")
        self.image_label_display = tk.Label(self.root, font=label_font, bg='gray')
        self.image_label_display.config(text="")
        self.image_label_display.place(x=rect_center_x, y=self.rect_end_y + 20, anchor="n")  # Adjust y as needed

        # Define element dimensions
        button_width = 80  # Approximate width for buttons
        gap = 10  # Gap between elements

        # Calculate the horizontal center of the whitespace to the right of the black rectangle
        rect_right_edge = self.rect_end_x
        screen_width = self.root.winfo_screenwidth()
        whitespace_midpoint = rect_right_edge + (screen_width - rect_right_edge) / 2

        # Place buttons
        prev_button_x = self.rect_start_x  # Align with the left side of the rectangle
        self.prev_button.place(x=prev_button_x, y=self.rect_end_y + gap, width=button_width)

        next_button_x = prev_button_x + button_width + gap  # To the right of the previous button with a gap
        self.next_button.place(x=next_button_x, y=self.rect_end_y + gap, width=button_width)

        select_button_width = 150  # Adjust based on the expected button width
        select_button_x = whitespace_midpoint - (select_button_width / 2)
        select_button_y = self.rect_start_y
        self.select_button.place(x=select_button_x, y=select_button_y, width=select_button_width)

        self.rgb_split_button.place(x=select_button_x, y=select_button_y + 3.5 * gap, width=select_button_width)

        self.red_channel_button.place(x=select_button_x, y=select_button_y + 7 * gap,
                                      width=select_button_width / 3)
        self.green_channel_button.place(x=select_button_x + select_button_width / 3, y=select_button_y + 7 * gap,
                                        width=select_button_width / 3)
        self.blue_channel_button.place(x=select_button_x + 2 * select_button_width / 3, y=select_button_y + 7 * gap,
                                       width=select_button_width / 3)
        self.rgb_restore_button.place(x=select_button_x, y=select_button_y + 10.5 * gap, width=select_button_width)
        self.drg_segment_button.place(x=select_button_x, y=select_button_y + 14 * gap, width=select_button_width)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelerApp(root)
    root.mainloop()
