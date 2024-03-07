import cv2
import display_functions as df
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np


def drg_segment(self):
    self.drg_segment_photo = None
    self.DRG_diameter.clear()

    def start_line(event):
        """Function to start drawing a line."""
        self.draw_start = (event.x, event.y)

    def draw_line(event):
        """Function to draw a line (temporary)."""
        if hasattr(self, 'temp_line'):
            self.drg_segment_canvas.delete(self.temp_line)
        self.temp_line = self.drg_segment_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x, event.y,
                                                             fill="white", width=3, tags="temp_line")

    def end_line(event):
        """Function to finalize drawing a line."""
        line_id = self.drg_segment_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x, event.y,
                                                      fill="white", width=3)
        self.DRG_line_ids.append((self.draw_start, (event.x, event.y), line_id))
        if hasattr(self, 'temp_line'):
            self.drg_segment_canvas.delete(self.temp_line)
            delattr(self, 'temp_line')

    def delete_line(event):
        """Function to delete a line."""
        proximity_threshold = 3  # Lower proximity threshold
        for start, end, line_id in self.DRG_line_ids[:]:
            if cv2.pointPolygonTest(np.array([start, end]), (event.x, event.y), True) >= -proximity_threshold:
                self.drg_segment_canvas.delete(line_id)
                self.DRG_line_ids.remove((start, end, line_id))
                break

    def apply_edge_detection(image):

        # Apply Gaussian Blur to smooth the image and reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, 50, 150)

        cv2.imshow('Debugging Window', edges)
        cv2.waitKey(0)  # Wait for a key press to proceed
        cv2.destroyAllWindows()  # Close the window
        return edges

    def find_ellipses(edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ellipses = []

        for contour in contours:
            if len(contour) >= 5:  # Minimum number of points for fitEllipse
                ellipse_segment = cv2.fitEllipse(contour)
                ellipses.append(ellipse_segment)

        return ellipses

    def segment_image():
        """Calculate lengths of all lines and display them."""
        if not self.DRG_line_ids:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires drawing at least 1 cell diameter")

        else:
            self.DRG_diameter = [np.linalg.norm(np.array(start) - np.array(end)) for start, end, _ in self.DRG_line_ids]
            # Clear lines information
            self.DRG_line_ids.clear()
            tolerance = 0.10  # Example: 50% tolerance
            min_diameter = np.min(self.DRG_diameter) * (1 - tolerance)
            max_diameter = np.max(self.DRG_diameter) * (1 + tolerance)

            image = self.gray_images[self.current_image_index]
            edges = apply_edge_detection(image)
            ellipses = find_ellipses(edges)
            filtered_ellipses = []
            for ellipse_segment in ellipses:
                axes = ellipse_segment[1]
                major_axis, minor_axis = axes
                ellipse_diameter = (major_axis + minor_axis) / 2

                if min_diameter <= ellipse_diameter <= max_diameter:
                    filtered_ellipses.append(ellipse_segment)

            for ellipse_segment in filtered_ellipses:
                center, axes, angle = ellipse_segment
                center = tuple(map(int, center))
                axes = tuple(map(int, axes))
                cv2.ellipse(image, center, axes, angle, 0, 360, (255, 0, 0), 2)  # Draw with green color

            # Convert the image with drawn ellipses to PhotoImage and display on canvas

            photo_image_with_ellipses = df.convert_to_photoimage(self, image)

            self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=photo_image_with_ellipses)

            # Keep a reference to the new PhotoImage.
            self.drg_segment_photo = photo_image_with_ellipses

    # Load the current image
    current_image = cv2.imread(self.image_file_paths[self.current_image_index])
    current_image = resize_image(self, current_image)

    # Split the image into its RGB channels
    r, g, b = cv2.split(current_image)

    channel = self.channel_var.get()

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)

    match channel:
        case 'Red':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = r

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(self, r)

        case 'Green':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = g

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(self, g)

        case 'Blue':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = b

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(self, b)

    # Display instruction message box
    messagebox.showinfo("Instructions", "Adjust contrast of image and draw cell diameters")

    # Create a new window for DRG Segmentation
    drg_segment_window = tk.Toplevel(self.root)
    drg_segment_window.title("DRG Segmentation")
    drg_segment_window.state('zoomed')

    # Display the image in the new window
    self.drg_segment_canvas = tk.Canvas(drg_segment_window, bg='white', width=photo.width(), height=photo.height())
    self.drg_segment_canvas_image_item = self.drg_segment_canvas.create_image(0, 0, anchor="nw", image=photo)
    self.drg_segment_canvas.pack()

    # Bind events for line drawing
    self.drg_segment_canvas.bind("<Button-1>", start_line)
    self.drg_segment_canvas.bind("<B1-Motion>", draw_line)
    self.drg_segment_canvas.bind("<ButtonRelease-1>", end_line)
    self.drg_segment_canvas.bind("<Button-3>", delete_line)  # Right-click to delete a line

    # Create finish button
    process_button = tk.Button(drg_segment_window, text="Process", command=lambda: segment_image())
    process_button.pack()
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

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        self.gray_images.append(gray_image)


def rgb_restore(self):
    load_and_resize_images(self)
    df.display_current_image(self)
