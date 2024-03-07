from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import os
import cv2
import numpy as np
import cv_functions as cvf


def rgb_split(self):
    if self.current_image_index is None or self.current_image_index < 0 or self.current_image_index >= len(
            self.image_file_paths):
        return  # Invalid index

    # Get the file path of the currently displayed image
    current_image = self.cv2_images[self.current_image_index]

    # Split the image into its RGB channels
    b, g, r = cv2.split(current_image)

    # Create separate RGB images for each channel
    r_rgb = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
    g_rgb = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    b_rgb = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])

    # Get image dimensions
    height, width, _ = r_rgb.shape

    # Get the window dimensions
    window_width = self.canvas.winfo_width()

    new_width = window_width / 3
    height_scale = new_width / width
    new_height = height * height_scale

    r_rgb_resized = cv2.resize(r_rgb, (int(new_width), int(new_height)),
                               interpolation=cv2.INTER_CUBIC)
    g_rgb_resized = cv2.resize(g_rgb, (int(new_width), int(new_height)),
                               interpolation=cv2.INTER_CUBIC)
    b_rgb_resized = cv2.resize(b_rgb, (int(new_width), int(new_height)),
                               interpolation=cv2.INTER_CUBIC)

    # Convert the RGB images to PhotoImage for displaying with Tkinter
    r_photo = convert_to_photoimage(self, r_rgb_resized)
    g_photo = convert_to_photoimage(self, g_rgb_resized)
    b_photo = convert_to_photoimage(self, b_rgb_resized)

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

    # Update window geometry to normalize it
    rgb_window.state('normal')

    # Run the Tkinter event loop
    rgb_window.mainloop()


def convert_to_photoimage(self, image):
    # Convert the image to PIL Image
    pil_image = Image.fromarray(image)
    # Convert the PIL Image to PhotoImage for displaying with Tkinter
    return ImageTk.PhotoImage(pil_image)


def position_image_in_canvas(self, image):
    # Get the dimensions of the resized image
    image_height, image_width, _ = image.shape

    # Create a canvas-sized image with the same dimensions as the canvas
    canvas_height = 10 * self.canvas.winfo_height()
    canvas_width = 10 * self.canvas.winfo_width()
    canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate the position to place the resized image within the canvas-sized image
    start_x = int(canvas_width / 2 - image_width / 2 + self.image_offset_x)
    start_y = int(canvas_height / 2 - image_height / 2 + self.image_offset_y)

    # Calculate the region where the resized image will be placed within the canvas-sized image
    end_x = start_x + image_width
    end_y = start_y + image_height

    # Place the resized image within the canvas-sized image
    canvas_image[start_y:end_y, start_x:end_x] = image

    return canvas_image


# Function to update the transparency mask and display the current image
def display_current_image(self):
    if not self.cv2_images:
        return  # No images to display

    self.canvas.delete("image")  # Clear previous image

    # Get the current image
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
    if self.image_labels:
        self.image_label_display.config(text=self.image_labels[self.current_image_index])


def prompt_for_labels(self):
    label_window = tk.Toplevel(self.root)
    label_window.title("Label Images")

    entries = []  # This will hold the entry widgets for labels
    for idx, path in enumerate(self.image_file_paths):
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
        display_current_image(self)
        update_image_label_display(self)

    save_button = tk.Button(label_window, text="Save Labels", command=save_labels)
    save_button.pack()


def next_image(self):
    if self.photoimages:
        self.current_image_index = (self.current_image_index + 1) % len(self.photoimages)
        self.image_offset_x = 0
        self.image_offset_y = 0
        display_current_image(self)
        update_image_label_display(self)  # Update the label display


def prev_image(self):
    if self.photoimages:
        self.current_image_index = (self.current_image_index - 1) % len(self.photoimages)
        self.image_offset_x = 0
        self.image_offset_y = 0
        display_current_image(self)
        update_image_label_display(self)  # Update the label display
