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

    # Resize images to fit the available space while maintaining aspect ratio
    r_rgb_resized = cvf.resize_image(self, r_rgb)
    g_rgb_resized = cvf.resize_image(self, g_rgb)
    b_rgb_resized = cvf.resize_image(self, b_rgb)

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

    # Update window geometry to maximize it
    rgb_window.state('normal')

    # Run the Tkinter event loop
    rgb_window.mainloop()


def convert_to_photoimage(self, image):
    # Convert the image to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Convert the PIL Image to PhotoImage for displaying with Tkinter
    return ImageTk.PhotoImage(pil_image)


def create_transparency_mask(window_size, rectangle_coords):
    width, height = window_size
    mask = np.full((height, width), 0, dtype=np.uint8)  # Initialize mask with fully opaque pixels
    x1, y1 = rectangle_coords[0]
    x2, y2 = rectangle_coords[1]
    mask[y1:y2, x1:x2] = 255  # Set pixels inside the rectangle to fully transparent
    return mask


# Function to apply transparency mask to the image
def apply_transparency(image, mask):
    # Resize the mask to match the dimensions of the image
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Stack the image and the resized mask along the third axis to create an RGBA image
    image_with_alpha = np.dstack((image, resized_mask))

    # Convert the RGBA image to PIL Image
    image_with_alpha_pil = Image.fromarray(image_with_alpha)

    return image_with_alpha_pil


# Function to update the transparency mask and display the current image
def display_current_image(self):
    if not self.photoimages:
        return  # No images to display

    self.canvas.delete("image")  # Clear previous image

    # Get the current image
    image = self.cv2_images[self.current_image_index]
    resized_image = cvf.resize_image(self, image)

    # Calculate the coordinates for placing the image centered within the black rectangle
    image_x = self.image_x - resized_image.shape[1] / 2
    image_y = self.image_y - resized_image.shape[0] / 2

    # Calculate the coordinates of the black rectangle relative to the window
    rectangle_coords = ((self.rect_start_x, self.rect_start_y), (self.rect_end_x, self.rect_end_y))

    # Get the size of the window
    window_size = (self.canvas.winfo_width(), self.canvas.winfo_height())

    # Create transparency mask
    transparency_mask = create_transparency_mask(window_size, rectangle_coords)

    # Apply transparency mask to the image
    image_with_transparency = apply_transparency(resized_image, transparency_mask)

    # Convert the image to PhotoImage for displaying with Tkinter
    photo_image = ImageTk.PhotoImage(image=image_with_transparency)

    self.photoimages[self.current_image_index] = photo_image

    # Display the image on the canvas
    self.canvas.create_image(image_x, image_y, anchor="nw", image=photo_image, tags="image")


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
        display_current_image(self)
        update_image_label_display(self)  # Update the label display


def prev_image(self):
    if self.photoimages:
        self.current_image_index = (self.current_image_index - 1) % len(self.photoimages)
        display_current_image(self)
        update_image_label_display(self)  # Update the label display


def photoimage_to_image(self, current_image):
    """
    Convert a PhotoImage object to a PIL Image object.
    """
    # Convert PhotoImage to PIL Image
    pil_image = Image.frombytes('RGB', (current_image.width(), current_image.height()), current_image.data)

    return pil_image
