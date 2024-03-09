import cv2
import display_functions as df
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from skimage import morphology
from skimage.measure import regionprops


def drg_segment(self):
    self.drg_segment_photo = None
    self.DRG_diameter.clear()

    def apply_contrast():
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
        gray = self.gray_images[self.current_image_index]
        enhanced = clahe.apply(gray)
        self.contrasted_gray_images[self.current_image_index] = enhanced
        contrasted_photo = df.convert_to_photoimage(self, enhanced)

        self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=contrasted_photo)

        self.drg_segment_photo = contrasted_photo

    def overlay_regions_as_white():
        """
        Modifies the original grayscale image to replace all pixels that are part of a region with white pixels.

        Parameters:
        - original_image: Grayscale image as a NumPy array.
        - regions: NumPy array of the same shape as original_image, with distinct regions labeled with unique integers.
                   Non-region areas should be labeled with 0.

        Returns:
        - A modified image where pixels in regions are white.
        """

        # Create a copy of the original image to avoid modifying it directly
        overlay_image = self.gray_images[self.current_image_index]

        # Identify all pixels that are part of any region (i.e., have a non-zero label in the regions array)
        region_pixels = self.regions[self.current_image_index] > 0

        # Replace these pixels with white (255 for a grayscale image)
        overlay_image[region_pixels] = 255

        overlay_color_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2BGR)
        props = regionprops(self.regions[self.current_image_index])

        for prop in props:
            # Skip background
            if prop.label == 0:
                continue

            # Get the centroid coordinates
            y, x = prop.centroid

            # Draw the region number at the centroid
            cv2.putText(overlay_color_image, str(prop.label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1)

        overlay_photo = df.convert_to_photoimage(self, overlay_color_image)
        self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=overlay_photo)

        self.drg_segment_photo = overlay_photo

    def grow_regions():
        if not self.process_statuses[self.current_image_index][0]:
            messagebox.showerror("Error", "Segmentation requires edge maps",
                                 parent=self.drg_segment_window)
        elif not self.process_statuses[self.current_image_index][1]:
            messagebox.showerror("Error", "Segmentation requires seeds",
                                 parent=self.drg_segment_window)
        elif not self.process_statuses[self.current_image_index][2]:
            messagebox.showerror("Error", "Segmentation requires diameters",
                                 parent=self.drg_segment_window)
        else:
            edge_map = self.edge_maps[self.current_image_index]
            mask = np.logical_not(edge_map)
            region_map = np.zeros(edge_map.shape, dtype=int)

            region_label = 1

            for coords, _ in self.seeds[self.current_image_index][:]:
                x = coords[0]
                y = coords[1]

                if edge_map[y, x] == 0:
                    region_map = morphology.flood_fill(region_map, (y, x), region_label, connectivity=1)
                    region_label += 1

            print(region_map)
            self.regions[self.current_image_index] = region_map
            overlay_regions_as_white()

    # Load the current image
    current_image = cv2.imread(self.image_file_paths[self.current_image_index])
    current_image = resize_image(self, current_image)

    # Split the image into its RGB channels
    r, g, b = cv2.split(current_image)

    channel = self.channel_var.get()

    match channel:
        case 'Red':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = r
            self.contrasted_gray_images[self.current_image_index] = r

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(self, r)

        case 'Green':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = g
            self.contrasted_gray_images[self.current_image_index] = g

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(self, g)

        case 'Blue':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = b
            self.contrasted_gray_images[self.current_image_index] = b

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(self, b)

    # Display instruction message box
    messagebox.showinfo("Instructions", "Follow segmentation steps in order starting with 1. Edges")
    self.drg_segment_photo = photo
    # Create a new window for DRG Segmentation
    self.drg_segment_window = tk.Toplevel(self.root)
    self.drg_segment_window.title("DRG Segmentation")
    self.drg_segment_window.state('zoomed')

    # Display the image in the new window
    self.drg_segment_canvas = tk.Canvas(self.drg_segment_window, bg='white', width=photo.width(), height=photo.height())
    self.drg_segment_canvas_image_item = self.drg_segment_canvas.create_image(0, 0, anchor="nw", image=photo)
    self.drg_segment_canvas.pack()

    # Create finish button
    edges_button = tk.Button(self.drg_segment_window, text="1. Edges", command=lambda: edge_detect(self))
    seeds_button = tk.Button(self.drg_segment_window, text="2. Set Seeds", command=lambda: set_seeds(self))
    lines_button = tk.Button(self.drg_segment_window, text="3. Draw cell diameters",
                             command=lambda: draw_diameters(self))
    regions_button = tk.Button(self.drg_segment_window, text="4. Grow cell regions", command=lambda: grow_regions())

    edges_button.pack()
    seeds_button.pack()
    lines_button.pack()
    regions_button.pack()

    apply_contrast()
    # Run the Tkinter event loop
    self.drg_segment_window.mainloop()


def draw_diameters(self):
    self.diameters_photo = None
    # Create a new window for DRG Segmentation
    diameters_window = tk.Toplevel(self.root)
    diameters_window.title("Draw diameters")
    diameters_window.state("zoomed")

    self.DRG_diameter.clear()

    def save_diameters():
        if not self.DRG_line_ids:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires drawing at least 1 cell diameter",
                                 parent=diameters_window)

        else:
            self.DRG_diameter = [np.linalg.norm(np.array(start) - np.array(end)) for start, end, _ in self.DRG_line_ids]
            # Clear lines information
            self.DRG_line_ids.clear()

            diameters_window.destroy()
            messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
            self.process_statuses[self.current_image_index][2] = True

    def start_line(event):
        """Function to start drawing a line."""
        self.draw_start = (event.x, event.y)

    def draw_line(event):
        """Function to draw a line (temporary)."""
        if hasattr(self, 'temp_line'):
            self.diameters_canvas.delete(self.temp_line)
        self.temp_line = self.diameters_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x, event.y,
                                                           fill="white", width=3, tags="temp_line")

    def end_line(event):
        """Function to finalize drawing a line."""
        line_id = self.diameters_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x, event.y,
                                                    fill="white", width=3)
        self.DRG_line_ids.append((self.draw_start, (event.x, event.y), line_id))
        if hasattr(self, 'temp_line'):
            self.diameters_canvas.delete(self.temp_line)
            delattr(self, 'temp_line')

    def delete_line(event):
        """Function to delete a line."""
        proximity_threshold = 30  # Lower proximity threshold
        for start, end, line_id in self.DRG_line_ids[:]:
            if cv2.pointPolygonTest(np.array([start, end]), (event.x, event.y), True) >= -proximity_threshold:
                self.diameters_canvas.delete(line_id)
                self.DRG_line_ids.remove((start, end, line_id))
                break

    initial_image = self.contrasted_gray_images[self.current_image_index]
    photo = df.convert_to_photoimage(self, initial_image)
    self.diameters_photo = photo

    self.diameters_canvas = tk.Canvas(diameters_window, bg='white', width=photo.width(),
                                      height=photo.height())
    self.diameters_canvas_image_item = self.diameters_canvas.create_image(0, 0, anchor="nw", image=photo)
    self.diameters_canvas.pack()

    save_button = tk.Button(diameters_window, text="Save", command=lambda: save_diameters())
    save_button.pack()

    # Bind events for line drawing
    diameters_window.bind("<Button-1>", start_line)
    diameters_window.bind("<B1-Motion>", draw_line)
    diameters_window.bind("<ButtonRelease-1>", end_line)
    diameters_window.bind("<Button-3>", delete_line)  # Right-click to delete a line

    diameters_window.mainloop()


def set_seeds(self):
    self.seeds_photo = None
    # Create a new window for DRG Segmentation
    seeds_window = tk.Toplevel(self.root)
    seeds_window.title("Set Seeds")
    seeds_window.state("zoomed")

    def save_seeds():
        if not self.seeds[self.current_image_index]:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires at least 1 seed", parent=seeds_window)

        else:
            seeds_window.destroy()
            messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
            self.process_statuses[self.current_image_index][1] = True

    def add_seed(event):
        seed_id = self.seeds_canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='white',
                                                width=2)
        self.seeds[self.current_image_index].append((np.array((event.x, event.y)), seed_id))

    def delete_seed(event):
        proximity_threshold = 30  # Lower proximity threshold
        for coords, seed_id in self.seeds[self.current_image_index][:]:
            dist = np.linalg.norm(coords - (event.x, event.y))
            if dist <= proximity_threshold:
                try:
                    self.seeds_canvas.delete(seed_id)
                    self.seeds.remove((coords, seed_id))
                    break
                except:
                    print("")

    initial_image = self.contrasted_gray_images[self.current_image_index]
    photo = df.convert_to_photoimage(self, initial_image)
    self.seeds_photo = photo

    self.seeds_canvas = tk.Canvas(seeds_window, bg='white', width=photo.width(),
                                  height=photo.height())
    self.seeds_canvas_image_item = self.seeds_canvas.create_image(0, 0, anchor="nw", image=photo)
    self.seeds_canvas.pack()

    save_button = tk.Button(seeds_window, text="Save", command=lambda: save_seeds())
    save_button.pack()

    self.seeds_canvas.bind("<Button-1>", add_seed)
    self.seeds_canvas.bind("<Button-3>", delete_seed)

    seeds_window.mainloop()


def edge_detect(self):
    # Create a new window for DRG Segmentation
    edges_window = tk.Toplevel(self.root)
    edges_window.title("Edge Detection Settings")

    initial_image = self.contrasted_gray_images[self.current_image_index]
    photo = df.convert_to_photoimage(self, initial_image)
    self.drg_segment_photo = photo
    self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=photo)

    def apply_edges(threshold1, threshold2, min_length, image):
        # Apply Gaussian Blur to smooth the image and reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, int(threshold1), int(threshold2))

        self.edge_maps[self.current_image_index] = edges
        edge_overlay = np.where(edges == 255, 255, image)
        # TODO: Filter by minimum length
        # TODO: Add in contour completion
        edge_photo = df.convert_to_photoimage(self, edge_overlay)

        self.drg_segment_photo = edge_photo
        self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=edge_photo)

    def save_edges():
        self.edge_thresholds[self.current_image_index][0] = t1_slider_value.get()
        self.edge_thresholds[self.current_image_index][1] = t2_slider_value.get()

        edges_window.destroy()
        messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
        self.process_statuses[self.current_image_index][0] = True

    apply_edges(50, 150, 20, initial_image)

    t1_slider_value = tk.IntVar(value=50)
    t2_slider_value = tk.IntVar(value=150)
    min_slider_value = tk.IntVar(value=20)

    t1_slider = tk.Scale(edges_window, from_=1, to=255, orient=tk.HORIZONTAL, variable=t1_slider_value,
                         label="Lower threshold", length=int(0.8 * photo.width()), width=20,
                         command=lambda value: apply_edges(value, t2_slider_value.get(), min_slider_value.get(),
                                                           initial_image))
    t2_slider = tk.Scale(edges_window, from_=1, to=255, orient=tk.HORIZONTAL, variable=t2_slider_value,
                         label="Upper threshold", length=int(0.8 * photo.width()), width=20,
                         command=lambda value: apply_edges(t1_slider_value.get(), value, min_slider_value.get(),
                                                           initial_image))
    min_slider = tk.Scale(edges_window, from_=0, to=150, orient=tk.HORIZONTAL, variable=min_slider_value,
                          label="Minimum length", length=int(0.5 * photo.width()), width=20,
                          command=lambda value: apply_edges(t1_slider_value.get(), t2_slider_value.get(), value,
                                                            initial_image))

    save_button = tk.Button(edges_window, text="Save", command=lambda: save_edges())

    t2_slider.pack()
    t1_slider.pack()
    min_slider.pack()
    save_button.pack()

    edges_window.mainloop()


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
        self.contrasted_gray_images.append(gray_image)
        self.edge_maps.append(gray_image)
        self.edge_thresholds.append([0, 0])
        self.seeds.append([])
        self.process_statuses.append([False, False, False])
        self.regions.append([])
