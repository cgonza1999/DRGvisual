import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont  # Import tkfont for custom font
import os
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImageLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')  # Maximize the main application window
        self.images = []  # Stores the PhotoImage objects for display
        self.image_file_paths = []  # Stores the file paths of the images
        self.image_labels = []  # Stores the labels for the images
        self.image_label_display = None  # Store the label widget for reference
        self.current_image_index = 0  # Tracks the current image
        self.setup_ui()
        self.max_width = None
        self.max_height = None

    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, bg='gray')  # Create a canvas with a gray background
        self.canvas.pack(fill='both', expand=True)  # Make the canvas fill the window

        # Coordinates for the black rectangle
        self.rect_start_x = 0.01 * self.root.winfo_screenwidth()
        self.rect_start_y = 0.01 * self.root.winfo_screenheight()
        self.rect_end_x = 0.76 * self.root.winfo_screenwidth()
        self.rect_end_y = 0.76 * self.root.winfo_screenheight()
        self.canvas.create_rectangle(self.rect_start_x, self.rect_start_y,
                                     self.rect_end_x, self.rect_end_y,
                                     fill="black", outline="black")

        # # Create a label widget for displaying image label
        label_font = tkfont.Font(family="Arial", size=14, weight="bold")
        self.image_label_display = tk.Label(self.root, font=label_font)
        self.update_image_label_display()  # Update label initially

        # Assuming you've calculated the rectangle's center x-coordinate
        rect_center_x = self.rect_start_x + (self.rect_end_x - self.rect_start_x) / 2

        # Since the exact width of the label text isn't known beforehand (varies with text),
        # you might place it first, then adjust its position if necessary.
        self.image_label_display.place(x=rect_center_x, y=self.rect_end_y + 20, anchor="n")  # Adjust y as needed

        # Wait for the window to be fully loaded to get accurate dimensions
        self.root.after(100, self.initialize_other_ui_elements)

    def update_image_label_display(self):
        # Update the label text here
        label_text = "Your updated label text"  # Replace this with the actual label text
        self.image_label_display.config(text=label_text)

        # Assuming you've calculated the rectangle's center x-coordinate
        rect_center_x = self.rect_start_x + (self.rect_end_x - self.rect_start_x) / 2

        # Place the label at the appropriate position
        if self.image_label_display:
            self.image_label_display.place_forget()  # Remove the old label
            self.image_label_display.place(x=rect_center_x, y=self.rect_end_y + 20, anchor="n")  # Adjust y as needed

    def initialize_other_ui_elements(self):
        # Calculate the y position slightly below the black rectangle
        ui_elements_y = self.rect_end_y + 10  # 10 pixels below the rectangle

        button_width = 80  # Approximate width for buttons
        gap = 10  # Gap between elements

        # Previous button
        self.prev_button = tk.Button(self.root, text="< Prev", command=self.prev_image)
        prev_button_x = self.rect_start_x  # Align with the left side of the rectangle
        self.prev_button.place(x=prev_button_x, y=ui_elements_y, width=button_width)

        # Next button
        self.next_button = tk.Button(self.root, text="Next >", command=self.next_image)
        next_button_x = prev_button_x + button_width + gap  # To the right of the previous button with a gap
        self.next_button.place(x=next_button_x, y=ui_elements_y, width=button_width)

        # Calculate the horizontal center of the whitespace to the right of the black rectangle
        rect_right_edge = self.rect_end_x
        screen_width = self.root.winfo_screenwidth()
        whitespace_midpoint = rect_right_edge + (screen_width - rect_right_edge) / 2

        # "Select Image Files" button
        select_button = tk.Button(self.root, text="Select Image Files", command=self.select_and_load_files)
        select_button_width = 150  # Adjust based on the expected button width
        select_button_x = whitespace_midpoint - (select_button_width / 2)

        # The top of the button aligned with the top of the black rectangle
        select_button_y = self.rect_start_y

        select_button.place(x=select_button_x, y=select_button_y, width=select_button_width)

        # RGB Split button
        rgb_split_button = tk.Button(self.root, text="RGB Split", command=self.rgb_split)

        rgb_split_button.place(x=select_button_x, y=select_button_y + 35, width=select_button_width)

    def rgb_split(self):
        if self.current_image_index is None or self.current_image_index < 0 or self.current_image_index >= len(
                self.image_file_paths):
            return  # Invalid index

        # Get the file path of the currently displayed image
        current_file_path = self.image_file_paths[self.current_image_index]

        # Read the image using OpenCV
        bgr_image = cv2.imread(current_file_path)

        # Split the image into its RGB channels
        b, g, r = cv2.split(bgr_image)

        # Create separate RGB images for each channel
        r_rgb = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
        g_rgb = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
        b_rgb = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])

        # Resize images to fit the available space while maintaining aspect ratio
        r_rgb_resized = self.resize_image(r_rgb)
        g_rgb_resized = self.resize_image(g_rgb)
        b_rgb_resized = self.resize_image(b_rgb)

        # Convert the RGB images to PhotoImage for displaying with Tkinter
        r_photo = self.convert_to_photoimage(r_rgb_resized)
        g_photo = self.convert_to_photoimage(g_rgb_resized)
        b_photo = self.convert_to_photoimage(b_rgb_resized)

        # Create a new window for displaying the RGB split
        rgb_window = tk.Toplevel(self.root)
        rgb_window.title("RGB Split")

        # Update window geometry to maximize it
        rgb_window.state('zoomed')

        # Display the RGB images side by side with labels expanding to fill the window
        r_label = tk.Label(rgb_window, image=r_photo)
        r_label.grid(row=0, column=0, sticky="nsew")  # Expand in all directions

        g_label = tk.Label(rgb_window, image=g_photo)
        g_label.grid(row=0, column=1, sticky="nsew")  # Expand in all directions

        b_label = tk.Label(rgb_window, image=b_photo)
        b_label.grid(row=0, column=2, sticky="nsew")  # Expand in all directions

        # Update window geometry to maximize it
        rgb_window.state('normal')

        # Run the Tkinter event loop
        rgb_window.mainloop()

    def resize_image(self, image):
        # Get image dimensions
        height, width, _ = image.shape

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Calculate target dimensions with padding
        padding = 10  # Adjust as needed
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        target_width = (window_width - 4 * padding) // 3
        target_height = window_height - 2 * padding

        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        # Check if the calculated dimensions exceed the available space
        if new_width > target_width or new_height > target_height:
            # If so, adjust the dimensions to fit within the available space
            scaling_factor = min(target_width / new_width, target_height / new_height)
            new_width = int(new_width * scaling_factor)
            new_height = int(new_height * scaling_factor)

        # Perform resizing
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return resized_image

    def convert_to_photoimage(self, image):
        # Convert the image to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Convert the PIL Image to PhotoImage for displaying with Tkinter
        return ImageTk.PhotoImage(pil_image)

    def select_and_load_files(self):
        file_types = [('Image files', '*.tiff;*.tif;*.jpg;*.jpeg;*.png;*.bmp'), ('All files', '*.*')]
        file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=file_types)
        if file_paths:
            self.image_file_paths = file_paths
            self.load_and_resize_images(file_paths)
            self.prompt_for_labels(file_paths)

    def load_and_resize_images(self, file_paths):
        self.images.clear()  # Clear existing images before loading new ones
        rect_width = self.rect_end_x - self.rect_start_x
        rect_height = self.rect_end_y - self.rect_start_y

        for file_path in file_paths:
            # Load the image using OpenCV
            cv_image = cv2.imread(file_path)
            # Convert color from BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Calculate the scaling factor
            img_height, img_width = cv_image.shape[:2]
            scale_width = rect_width / img_width
            scale_height = rect_height / img_height
            scale_factor = min(scale_width, scale_height)

            # Resize the image
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            resized_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to PIL image and then to PhotoImage for Tkinter
            pil_image = Image.fromarray(resized_image)
            photo_image = ImageTk.PhotoImage(pil_image)
            self.images.append(photo_image)

    def prompt_for_labels(self, file_paths):
        label_window = tk.Toplevel(self.root)
        label_window.title("Label Images")

        entries = []  # This will hold the entry widgets for labels
        for idx, path in enumerate(file_paths):
            row = tk.Frame(label_window)
            row.pack(fill='x')

            tk.Label(row, text=f"Label for image {idx + 1}:").pack(side='left')

            entry = tk.Entry(row)
            entry.pack(side='right', expand=True, fill='x')
            entry.insert(0, os.path.basename(path))  # Default label is the basename of the file path
            entries.append(entry)

        def save_labels():
            # Directly use 'entries' and 'label_window' from the enclosing scope
            self.image_labels = [entry.get() for entry in entries]
            print("Labels saved:", self.image_labels)  # Placeholder for demonstration
            label_window.destroy()
            self.current_image_index = 0  # Assuming you want to display the first image and its label
            self.display_current_image()
            self.update_image_label_display()

        save_button = tk.Button(label_window, text="Save Labels", command=save_labels)
        save_button.pack()

    def update_image_label_display(self):
        if self.image_labels:
            self.image_label_display.config(text=self.image_labels[self.current_image_index])

    def next_image(self):
        if self.images:
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            self.display_current_image()
            self.update_image_label_display()  # Update the label display

    def prev_image(self):
        if self.images:
            self.current_image_index = (self.current_image_index - 1) % len(self.images)
            self.display_current_image()
            self.update_image_label_display()  # Update the label display

    def display_current_image(self):
        if not self.images:
            return  # No images to display

        self.canvas.delete("image")  # Clear previous image

        # Get the current image
        image = self.images[self.current_image_index]

        # Dimensions of the black rectangle
        rect_width = self.rect_end_x - self.rect_start_x
        rect_height = self.rect_end_y - self.rect_start_y

        # Calculate the center position for the black rectangle
        rect_center_x = self.rect_start_x + rect_width / 2
        rect_center_y = self.rect_start_y + rect_height / 2

        # Calculate the position to place the image such that it's centered within the black rectangle
        image_x = rect_center_x - image.width() / 2
        image_y = rect_center_y - image.height() / 2

        # Display the image, centered within the black rectangle
        self.canvas.create_image(image_x, image_y, anchor="nw", image=image, tags="image")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelerApp(root)
    root.mainloop()
