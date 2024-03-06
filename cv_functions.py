import cv2
import display_functions as df
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np


def isolate_channel(self, channel):
    if self.current_image_index is None:
        return

    for idx in range(0, len(self.cv2_images)):
        current_image = self.cv2_images[idx]
        current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)

        # Split the image into its RGB channels
        b, g, r = cv2.split(current_image)

        # Create separate RGB images for each channel
        r_rgb = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
        g_rgb = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
        b_rgb = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])

        r_rgb = cv2.cvtColor(r_rgb, cv2.COLOR_BGR2RGB)
        g_rgb = cv2.cvtColor(g_rgb, cv2.COLOR_BGR2RGB)
        b_rgb = cv2.cvtColor(b_rgb, cv2.COLOR_BGR2RGB)

        # Convert the RGB images to PhotoImage for displaying with Tkinter
        r_photo = df.convert_to_photoimage(self, r_rgb)
        g_photo = df.convert_to_photoimage(self, g_rgb)
        b_photo = df.convert_to_photoimage(self, b_rgb)

        match channel:
            case "r":
                # Update cv2_images with the isolated green channel
                self.cv2_images[idx] = r_rgb
                self.photoimages[idx] = r_photo

            case "g":
                # Update cv2_images with the isolated green channel
                self.cv2_images[idx] = g_rgb
                self.photoimages[idx] = g_photo

            case "b":
                # Update cv2_images with the isolated green channel
                self.cv2_images[idx] = b_rgb
                self.photoimages[idx] = b_photo

    df.display_current_image(self)


def drg_segment(self):
    # Function to start drawing a line
    def start_line(event):
        self.DRG_line_coords.append((event.x, event.y))
        canvas.bind("<Motion>", draw_line)

    # Function to draw a line
    def draw_line(event):
        canvas.delete("temp_line")
        canvas.create_line(self.DRG_line_coords[-1][0], self.DRG_line_coords[-1][1], event.x, event.y, fill="white",
                           width=3, tags="temp_line")

    # Function to end drawing a line
    def end_line(event):
        self.DRG_line_coords.append((event.x, event.y))
        canvas.create_line(self.DRG_line_coords[-2][0], self.DRG_line_coords[-2][1], self.DRG_line_coords[-1][0],
                           self.DRG_line_coords[-1][1], fill="white", width=3, tags="line")
        canvas.unbind("<Motion>")

    def finish_drawing():
        # Placeholder for saving the drawn lines or performing any other action
        messagebox.showinfo("Finish Drawing", "Drawing finished!")

    # Load the current image
    current_image = cv2.imread(self.image_file_paths[self.current_image_index])
    current_image = resize_image(self, current_image)

    # Split the image into its RGB channels
    b, g, r = cv2.split(current_image)

    # Create separate RGB images for each channel
    g_rgb = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    g_rgb = cv2.cvtColor(g_rgb, cv2.COLOR_BGR2RGB)

    # Convert the image to PhotoImage for displaying with Tkinter
    g_photo = df.convert_to_photoimage(self, g_rgb)

    # Display instruction message box
    messagebox.showinfo("Instructions", "Please draw lines across cell radii.")

    # Create a new window for DRG Segmentation
    drg_segment_window = tk.Toplevel(self.root)
    drg_segment_window.title("DRG Segmentation")

    # Display the image in the new window
    canvas = tk.Canvas(drg_segment_window, bg='white', width=g_photo.width(), height=g_photo.height())
    canvas.create_image(0, 0, anchor="nw", image=g_photo)
    canvas.pack()

    # Bind events for line drawing
    canvas.bind("<Button-1>", start_line)
    canvas.bind("<ButtonRelease-1>", end_line)

    # Create finish button
    finish_button = tk.Button(drg_segment_window, text="Finish", command=finish_drawing)
    finish_button.pack()
    # Run the Tkinter event loop
    drg_segment_window.mainloop()


def resize_image(self, image):
    # Get image dimensions
    height, width, _ = image.shape

    # Get the window dimensions
    window_width = self.rect_width
    window_height = self.rect_height

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Calculate the target dimensions while maintaining aspect ratio
    if window_width / window_height > aspect_ratio:
        new_width = window_height * aspect_ratio
        new_height = window_height
    else:
        new_width = window_width
        new_height = window_width / aspect_ratio

    # Perform resizing
    resized_image = cv2.resize(image, (int(new_width), int(new_height)),
                               interpolation=cv2.INTER_CUBIC)
    return resized_image


def select_and_load_files(self):
    file_types = [('Image files', '*.tiff;*.tif;*.jpg;*.jpeg;*.png;*.bmp'), ('All files', '*.*')]
    file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=file_types)
    if file_paths:
        self.image_file_paths = file_paths
        load_and_resize_images(self)
        df.prompt_for_labels(self)


def load_and_resize_images(self):
    self.photoimages.clear()  # Clear existing images before loading new ones
    self.cv2_images.clear()  # Clear existing images before loading new ones
    self.image_offset_x = 0
    self.image_offset_y = 0

    for file_path in self.image_file_paths:
        # Load the image using OpenCV
        cv_image = cv2.imread(file_path)

        # Convert color from BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        resized_image = resize_image(self, cv_image)
        self.cv2_images.append(resized_image)

        # Convert to PIL image and then to PhotoImage for Tkinter
        photo_image = df.convert_to_photoimage(self, resized_image)
        self.photoimages.append(photo_image)


def rgb_restore(self):
    load_and_resize_images(self)
    df.display_current_image(self)
