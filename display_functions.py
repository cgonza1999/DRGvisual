from PIL import Image, ImageTk
import tkinter as tk
import os
import cv2
import numpy as np


# Image Display and Manipulation Functions

def convert_to_photoimage(image):
    """
    Convert a CV2 image to a Tkinter PhotoImage object.

    :param image: The CV2 image to convert.
    :return: The Tkinter PhotoImage object.
    """
    pil_image = Image.fromarray(image)
    return ImageTk.PhotoImage(pil_image)


def position_image_in_canvas(self, image):
    """
    Position and resize an image to fit within the application's canvas.

    :param image: The CV2 image to be positioned.
    :return: A new CV2 image positioned within a larger canvas-sized image.
    """
    # Calculate dimensions and position
    image_height, image_width, _ = image.shape
    canvas_height = 10 * self.canvas.winfo_height()
    canvas_width = 10 * self.canvas.winfo_width()
    start_x = int(canvas_width / 2 - image_width / 2 + self.image_offset_x)
    start_y = int(canvas_height / 2 - image_height / 2 + self.image_offset_y)

    # Create and place the image
    canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    end_x = start_x + image_width
    end_y = start_y + image_height
    canvas_image[start_y:end_y, start_x:end_x] = image

    return canvas_image


def display_current_image(self):
    """
    Display the current image on the application's canvas.
    """
    if not self.cv2_images:
        return

    self.canvas.delete("all")  # Clear the canvas before displaying a new image
    image = self.cv2_images[self.current_image_index]
    canvas_image = position_image_in_canvas(self, image)

    # Crop the canvas image
    crop_x_start = int(9 / 20 * canvas_image.shape[0])
    crop_x_end = int(11 / 20 * canvas_image.shape[0])
    crop_y_start = int(9 / 20 * canvas_image.shape[1])
    crop_y_end = int(11 / 20 * canvas_image.shape[1])
    cropped_canvas_image = canvas_image[crop_x_start:crop_x_end, crop_y_start:crop_y_end]

    # Create a mask for pixels outside the bounding box
    mask = np.zeros((self.canvas.winfo_height(), self.canvas.winfo_width()), dtype=np.uint8)
    mask[self.rect_start_y:self.rect_end_y, self.rect_start_x:self.rect_end_x] = 255

    # Create an alpha channel with max opacity inside the bounding box
    alpha_map = np.where(mask == 255, 255, 0).astype(np.uint8)  # Ensure the correct data type

    # Convert the image to PhotoImage for displaying with Tkinter
    pil_image = Image.fromarray(cropped_canvas_image)

    # Apply the alpha channel to the image
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')  # Convert to RGBA mode if not already
    pil_image.putalpha(Image.fromarray(alpha_map))

    photo_image = ImageTk.PhotoImage(pil_image)

    self.photoimages[self.current_image_index] = photo_image

    # Display the image on the canvas
    self.canvas.create_image(0, 0, anchor="nw", image=photo_image, tags="image")
def update_image_label_display(self):
    """
    Update the display of the current image's label.
    """
    if self.image_labels:
        self.image_label_display.config(text=self.image_labels[self.current_image_index])


# Image Analysis and Transformation Functions

def rgb_split(self):
    """
    Split the current image into its RGB components and display them.
    """
    if self.current_image_index is None:
        return

    current_image = self.cv2_images[self.current_image_index]
    r, g, b = cv2.split(current_image)
    r_rgb = cv2.merge([r, np.zeros_like(g), np.zeros_like(b)])
    g_rgb = cv2.merge([np.zeros_like(r), g, np.zeros_like(b)])
    b_rgb = cv2.merge([np.zeros_like(r), np.zeros_like(g), b])

    # Resize for display
    window_width = self.canvas.winfo_width() / 3
    resized_images = [cv2.resize(img, (int(window_width), int(window_width * img.shape[0] / img.shape[1])),
                                 interpolation=cv2.INTER_CUBIC) for img in [r_rgb, g_rgb, b_rgb]]

    # Convert to PhotoImage and display
    photos = [convert_to_photoimage(img) for img in resized_images]
    rgb_window = tk.Toplevel(self.root)
    rgb_window.title("RGB Split")
    rgb_window.state('zoomed')

    for i, photo in enumerate(photos):
        tk.Label(rgb_window, image=photo).grid(row=0, column=i, sticky="nsew")

    rgb_window.state('normal')
    rgb_window.mainloop()


# Navigation Functions

def next_image(self):
    """
    Display the next image in the series.
    """
    if self.current_image_index is not None:
        self.current_image_index = (self.current_image_index + 1) % len(self.cv2_images)
        display_current_image(self)
        update_image_label_display(self)


def prev_image(self):
    """
    Display the previous image in the series.
    """
    if self.current_image_index is not None:
        self.current_image_index = (self.current_image_index - 1) % len(self.cv2_images)
        display_current_image(self)
        update_image_label_display(self)


# Image Labeling Functions

def prompt_for_labels(self):
    """
    Prompt the user to label the loaded images.
    """
    label_window = tk.Toplevel(self.root)
    label_window.title("Label Images")
    entries = []

    for idx, path in enumerate(self.image_file_paths):
        row = tk.Frame(label_window)
        row.pack(fill='x')
        tk.Label(row, text=f"Label for image {idx + 1}:").pack(side='left')

        entry = tk.Entry(row)
        entry.pack(side='right', expand=True, fill='x')
        entry.insert(0, os.path.basename(path))
        entries.append(entry)

    def save_labels():
        self.image_labels = [entry.get() for entry in entries]
        label_window.destroy()
        if self.cv2_images:
            self.current_image_index = 0
            display_current_image(self)
            update_image_label_display(self)

    tk.Button(label_window, text="Save Labels", command=save_labels).pack()
