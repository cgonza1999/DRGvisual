import cv2
import display_functions as df
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from skimage.draw import polygon
from skimage import filters


def drg_segment(self):
    self.drg_segment_photo = None
    self.drg_segment_image = None

    def apply_contrast():
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
        gray = self.gray_images[self.current_image_index]
        enhanced = clahe.apply(gray)
        self.contrasted_gray_images[self.current_image_index] = enhanced
        contrasted_photo = df.convert_to_photoimage(enhanced)

        self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=contrasted_photo)

        self.drg_segment_photo = contrasted_photo

    def targeted_smoothing(projections, window_size=3, variance_threshold=10):
        """
        Smooth areas with high variance in the projection lengths and preserve low variance areas.

        :param projections: List or numpy array of projection lengths.
        :param window_size: The size of the window to calculate local variance.
        :param variance_threshold: Threshold of variance to start applying smoothing.
        :return: Smoothed list of projection lengths.
        """
        length = len(projections)
        smoothed_projections = np.copy(projections)
        half_window = window_size // 2

        for i in range(length):
            start_index = max(0, i - half_window)
            end_index = min(length, i + half_window + 1)
            window = projections[start_index:end_index]

            # Calculate local variance in the window
            local_variance = np.var(window)

            # Check if the local variance exceeds the threshold
            if local_variance > variance_threshold:
                # Apply smoothing for high variance regions
                local_mean = np.mean(window)
                smoothed_projections[i] = local_mean
            # Else, leave the projection as is for low variance regions

        return smoothed_projections

    def radial_projection_with_adjustment(edge_map, seed, angle_step=1):
        projections = []
        angles = np.arange(0, 360, angle_step)

        max_length = int(np.max(self.DRG_diameter[self.current_image_index]) / 2)
        # Calculate projections
        for angle in angles:
            for length in range(1, max_length + 1):
                dx = int(length * np.cos(np.radians(angle)))
                dy = int(length * np.sin(np.radians(angle)))
                x, y = seed[0] + dx, seed[1] + dy
                if not (0 <= x < edge_map.shape[1] and 0 <= y < edge_map.shape[0]) or edge_map[
                    y, x] == 255 or length == max_length:
                    projections.append(length)
                    break

        for i in range(0, len(projections)):
            if projections[i] == max_length:
                projections[i] = np.median(projections[i - 3:i - 1])

        return targeted_smoothing(projections)

    def generate_roi_from_projections(edge_map, seed, adjusted_projections, angle_step=1):
        angles = np.arange(0, 360, angle_step)
        polygon_points_x = []
        polygon_points_y = []

        for i, length in enumerate(adjusted_projections):
            angle = np.radians(angles[i])
            x = seed[0] + length * np.cos(angle)
            y = seed[1] + length * np.sin(angle)
            polygon_points_x.append(x)
            polygon_points_y.append(y)

        rr, cc = polygon(polygon_points_y, polygon_points_x, shape=edge_map.shape)
        roi_map = np.zeros(edge_map.shape, dtype=np.uint8)
        roi_map[rr, cc] = 1  # Fill the polygon to generate ROI
        return roi_map

    def grow_regions():
        if not all(self.process_statuses[self.current_image_index]):
            messagebox.showerror("Error", "Segmentation requires edge maps, seeds, and diameters",
                                 parent=self.drg_segment_window)
            return

        edge_map = self.edge_maps[self.current_image_index]
        composite_roi_map = np.zeros(edge_map.shape, dtype=np.uint8)
        base_image = self.gray_images[self.current_image_index]
        contrasted_image = self.contrasted_gray_images[self.current_image_index]
        for seed, _ in self.seeds[self.current_image_index][:]:
            adjusted_projections = radial_projection_with_adjustment(edge_map, seed)
            roi_map = generate_roi_from_projections(edge_map, seed, adjusted_projections)
            composite_roi_map = np.logical_or(composite_roi_map, roi_map).astype(np.uint8)

            masked_region = np.where(roi_map != 0, base_image, 0)
            isodata_threshold = filters.threshold_isodata(masked_region[masked_region > 0])
            isodata_region = np.where(masked_region >= isodata_threshold, base_image, 0)
            self.positive_areas[self.current_image_index].append(np.count_nonzero(isodata_region))
            self.positive_intensities[self.current_image_index].append(np.mean(isodata_region[isodata_region != 0]))

        base_image_roi = np.where(composite_roi_map == 1, 255, contrasted_image)

        inverse_masked_region = np.where(composite_roi_map == 0, base_image, 0)

        li_threshold = filters.threshold_li(inverse_masked_region[inverse_masked_region > 0])
        li_region = np.where(inverse_masked_region >= li_threshold, base_image, 0)
        self.background_intensities[self.current_image_index] = np.mean(li_region[li_region != 0])

        for i in range(0, len(self.positive_areas[self.current_image_index])):
            intensity_diff = self.positive_intensities[self.current_image_index][i] - self.background_intensities[
                self.current_image_index]
            self.ctcf[self.current_image_index].append(
                intensity_diff * self.positive_areas[self.current_image_index][i])

        edge_photo = df.convert_to_photoimage(base_image_roi)

        self.drg_segment_photo = edge_photo

        self.drg_segment_canvas.create_image(0, 0, anchor="nw", image=edge_photo)

    # Load the current image
    current_image = cv2.imread(self.image_file_paths[self.current_image_index])
    current_image = resize_image(self, current_image)

    # Split the image into its RGB channels
    r, g, b = cv2.split(current_image)

    channel = self.channel_var.get()

    # Convert the image to PhotoImage for displaying with Tkinter
    photo = df.convert_to_photoimage(r)

    match channel:
        case 'Red':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = r
            self.contrasted_gray_images[self.current_image_index] = r

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(r)

        case 'Green':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = g
            self.contrasted_gray_images[self.current_image_index] = g

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(g)

        case 'Blue':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = b
            self.contrasted_gray_images[self.current_image_index] = b

            # Convert the image to PhotoImage for displaying with Tkinter
            photo = df.convert_to_photoimage(b)

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
    # Create a new window for DRG Segmentation
    diameters_window = tk.Toplevel(self.root)
    diameters_window.title("Draw diameters")

    self.draw_start = None

    self.DRG_diameter[self.current_image_index] = []

    def save_diameters():
        if not self.DRG_line_ids:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires drawing at least 1 cell diameter",
                                 parent=diameters_window)

        else:
            self.DRG_diameter[self.current_image_index] = [np.linalg.norm(np.array(start) - np.array(end)) for
                                                           start, end, _ in self.DRG_line_ids]
            # Clear lines information
            self.DRG_line_ids.clear()

            diameters_window.destroy()
            messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
            self.process_statuses[self.current_image_index][2] = True

            self.drg_segment_canvas.unbind("<Button-1>")
            self.drg_segment_canvas.unbind("<B1-Motion>")
            self.drg_segment_canvas.unbind("<ButtonRelease-1>")
            self.drg_segment_canvas.unbind("<Button-3>")  # Right-click to delete a line

    def start_line(event):
        """Function to start drawing a line."""
        self.draw_start = (event.x, event.y)

    def draw_line(event):
        """Function to draw a line (temporary)."""
        if hasattr(self, 'temp_line'):
            self.drg_segment_canvas.delete(self.temp_line)
        if self.draw_start:
            self.temp_line = self.drg_segment_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x,
                                                                 event.y,
                                                                 fill="white", width=3, tags="temp_line")

    def end_line(event):
        """Function to finalize drawing a line."""
        if self.draw_start:
            line_id = self.drg_segment_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x, event.y,
                                                          fill="white", width=3)
            self.DRG_line_ids.append((self.draw_start, (event.x, event.y), line_id))
            if hasattr(self, 'temp_line'):
                self.drg_segment_canvas.delete(self.temp_line)
                delattr(self, 'temp_line')

    def delete_line(event):
        """Function to delete a line."""
        proximity_threshold = 30  # Lower proximity threshold
        for start, end, line_id in self.DRG_line_ids[:]:
            if cv2.pointPolygonTest(np.array([start, end]), (event.x, event.y), True) >= -proximity_threshold:
                self.drg_segment_canvas.delete(line_id)
                self.DRG_line_ids.remove((start, end, line_id))
                break

    save_button = tk.Button(diameters_window, text="Save", command=lambda: save_diameters())
    save_button.pack()

    # Bind events for line drawing
    self.drg_segment_canvas.bind("<Button-1>", start_line)
    self.drg_segment_canvas.bind("<B1-Motion>", draw_line)
    self.drg_segment_canvas.bind("<ButtonRelease-1>", end_line)
    self.drg_segment_canvas.bind("<Button-3>", delete_line)  # Right-click to delete a line

    diameters_window.mainloop()


def set_seeds(self):
    # Create a new window for DRG Segmentation
    seeds_window = tk.Toplevel(self.root)
    seeds_window.title("Set Seeds")

    def save_seeds():
        if not self.seeds[self.current_image_index]:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires at least 1 seed", parent=seeds_window)

        else:
            seeds_window.destroy()
            messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
            self.process_statuses[self.current_image_index][1] = True
            self.drg_segment_canvas.unbind("<Button-1>")
            self.drg_segment_canvas.unbind("<Button-3>")

    def add_seed(event):
        seed_id = self.drg_segment_canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='white',
                                                      width=2)
        self.seeds[self.current_image_index].append((np.array((event.x, event.y)), seed_id))

    def delete_seed(event):
        proximity_threshold = 30  # Lower proximity threshold
        for coords, seed_id in self.seeds[self.current_image_index][:]:
            dist = np.linalg.norm(coords - (event.x, event.y))
            if dist <= proximity_threshold:
                try:
                    self.drg_segment_canvas.delete(seed_id)
                    self.seeds.remove((coords, seed_id))
                    break
                except:
                    print("")

    save_button = tk.Button(seeds_window, text="Save", command=lambda: save_seeds())
    save_button.pack()

    self.drg_segment_canvas.bind("<Button-1>", add_seed)
    self.drg_segment_canvas.bind("<Button-3>", delete_seed)

    seeds_window.mainloop()


def edge_detect(self):
    # Create a new window for DRG Segmentation
    edges_window = tk.Toplevel(self.root)
    edges_window.attributes('-topmost', True)
    edges_window.title("Edge Detection Settings")

    initial_image = self.contrasted_gray_images[self.current_image_index]
    photo = df.convert_to_photoimage(initial_image)
    self.drg_segment_image = initial_image
    self.drg_segment_photo = photo
    self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=photo)

    def apply_edges(threshold1, threshold2, min_length, image, event=None):
        # Initialize or update the edge map
        if event is None:
            # Your existing edge application logic
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred_image, int(threshold1), int(threshold2))

            # Find contours and filter based on length
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_map = np.zeros_like(edges)

            for contour in contours:
                if cv2.arcLength(contour, closed=True) >= int(min_length):
                    cv2.drawContours(filtered_map, [contour], -1, 255, thickness=cv2.FILLED)

            # Update the display with filtered edge map
            edge_overlay = np.where(filtered_map == 255, 255, image)
            edge_photo = df.convert_to_photoimage(edge_overlay)
            self.edge_maps[self.current_image_index] = filtered_map
            self.drg_segment_photo = edge_photo
            self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=edge_photo)
        else:
            # Contour deletion logic with nearest contour finding integrated
            contours, _ = cv2.findContours(self.edge_maps[self.current_image_index].copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            click_pos = np.array([event.x, event.y])
            nearest_contour_index = None
            min_distance = np.inf

            for i, contour in enumerate(contours):
                # Calculate the minimum distance from the click to this contour
                distances = np.sqrt(((contour - click_pos) ** 2).sum(axis=2))
                min_contour_distance = np.min(distances)
                if min_contour_distance < min_distance:
                    min_distance = min_contour_distance
                    nearest_contour_index = i

            # If a nearest contour is found and within deletion threshold, delete it
            if nearest_contour_index is not None:
                cv2.drawContours(self.edge_maps[self.current_image_index], [contours[nearest_contour_index]], -1, 0,
                                 thickness=-1)

            # Refresh the display after contour deletion
            edge_overlay = np.where(self.edge_maps[self.current_image_index] == 255, 255, image)
            edge_photo = df.convert_to_photoimage(edge_overlay)
            self.drg_segment_photo = edge_photo
            self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=edge_photo)

    def save_edges():
        self.edge_thresholds[self.current_image_index][0] = t1_slider_value.get()
        self.edge_thresholds[self.current_image_index][1] = t2_slider_value.get()

        edges_window.destroy()
        messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
        self.process_statuses[self.current_image_index][0] = True
        self.drg_segment_canvas.unbind("<Button-3>")

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

    self.drg_segment_canvas.bind("<Button-3>",
                                 lambda event: apply_edges(t1_slider_value.get(), t2_slider_value.get(),
                                                           min_slider_value.get(), initial_image, event))
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
        photo_image = df.convert_to_photoimage(resized_image)
        self.photoimages.append(photo_image)

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        self.gray_images.append(gray_image)
        self.contrasted_gray_images.append(gray_image)
        self.DRG_diameter.append([])
        self.edge_maps.append(gray_image)
        self.edge_thresholds.append([0, 0])
        self.seeds.append([])
        self.process_statuses.append([False, False, False])
        self.regions.append([])
        self.positive_areas.append([])
        self.positive_intensities.append([])
        self.background_intensities.append(0)
        self.ctcf.append([])
