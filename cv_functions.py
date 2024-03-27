import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from skimage.draw import polygon
from skimage import filters
import display_functions as df
import pandas as pd
from datetime import datetime


# Image Segmentation and Analysis Functions

def drg_segment(self):
    """
    Perform the segmentation process for Dorsal Root Ganglion (DRG) analysis.
    """

    self.drg_segment_photo = None
    self.roi_photo = None
    self.drg_segment_image = None

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
            single_channel_im = cv2.merge([r, np.zeros_like(g), np.zeros_like(b)])

        case 'Green':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = g
            self.contrasted_gray_images[self.current_image_index] = g

            # Convert the image to PhotoImage for displaying with Tkinter
            single_channel_im = cv2.merge([np.zeros_like(r), g, np.zeros_like(b)])

        case 'Blue':
            # Create separate RGB images for each channel
            self.gray_images[self.current_image_index] = b
            self.contrasted_gray_images[self.current_image_index] = b

            single_channel_im = cv2.merge([np.zeros_like(r), np.zeros_like(g), b])

    # Convert the image to PhotoImage for displaying with Tkinter
    photo = df.convert_to_photoimage(single_channel_im)

    # Display instruction message box
    messagebox.showinfo("Instructions", "Follow segmentation steps in order starting with 1. Edges")
    self.drg_segment_photo = photo
    # Create a new window for DRG Segmentation
    self.drg_segment_window = tk.Toplevel(self.root)
    self.drg_segment_window.title("DRG Segmentation")
    self.drg_segment_window.state('zoomed')

    # Display the image in the new window
    self.drg_segment_canvas = tk.Canvas(self.drg_segment_window, bg='white', width=photo.width(),
                                        height=photo.height())
    self.drg_segment_canvas_image_item = self.drg_segment_canvas.create_image(0, 0, anchor="nw", image=photo)
    self.drg_segment_canvas.pack()

    # Create finish button
    self.edges_button = tk.Button(self.drg_segment_window, text="1. Edges", command=lambda: edge_detect(self))
    self.seeds_button = tk.Button(self.drg_segment_window, text="2. Set Seeds", command=lambda: set_seeds(self))
    self.lines_button = tk.Button(self.drg_segment_window, text="3. Draw cell diameters",
                                  command=lambda: draw_diameters(self))
    self.regions_button = tk.Button(self.drg_segment_window, text="4. Grow cell regions",
                                    command=lambda: grow_regions(self))
    # Create the "Save" button
    self.save_data_button = tk.Button(self.drg_segment_window, text="Save",
                                      command=lambda: save_current_image_data(self))

    self.edges_button.pack()
    self.seeds_button.pack()
    self.lines_button.pack()
    self.regions_button.pack()

    apply_contrast(self)

    # Define the Treeview (table) and its columns
    self.info_table = tk.ttk.Treeview(self.drg_segment_window)
    self.info_table["columns"] = ("Label", "Seed#", "Seed pos", "ROI area", "CTCF")
    self.info_table.column("#0", width=0, stretch=tk.NO)  # Phantom column
    self.info_table.column("Label", anchor=tk.W, width=60)
    self.info_table.column("Seed#", anchor=tk.W, width=25)
    self.info_table.column("Seed pos", anchor=tk.W, width=50)
    self.info_table.column("ROI area", anchor=tk.W, width=40)
    self.info_table.column("CTCF", anchor=tk.W, width=40)

    # Define the headings
    self.info_table.heading("#0", text="", anchor=tk.W)
    self.info_table.heading("Label", text="Label", anchor=tk.W)
    self.info_table.heading("Seed#", text="Seed#", anchor=tk.W)
    self.info_table.heading("Seed pos", text="Seed pos", anchor=tk.W)
    self.info_table.heading("ROI area", text="ROI area", anchor=tk.W)
    self.info_table.heading("CTCF", text="CTCF", anchor=tk.W)

    # Position the table next to the image canvas
    self.info_table.place(anchor='ne', x=self.root.winfo_screenwidth(), y=0)
    self.drg_segment_window.update_idletasks()
    self.save_data_button.place(anchor='nw', x=self.info_table.winfo_x(),
                                y=self.info_table.winfo_height() + 10)

    self.save_data_button["state"] = "disabled"
    # Run the Tkinter event loop
    self.drg_segment_window.mainloop()

    # Additional setup can be added here as needed.


def grow_regions(self):
    """
    Grow the regions for segmentation based on the seeds and projections.
    """
    if not all(self.process_statuses[self.current_image_index]):
        messagebox.showerror("Error", "Segmentation requires edge maps, seeds, and diameters",
                             parent=self.drg_segment_window)
        return

    edge_map = self.edge_maps[self.current_image_index]
    composite_roi_map = np.zeros(edge_map.shape, dtype=np.uint8)
    base_image = self.gray_images[self.current_image_index]
    roi_overlay = cv2.merge([np.zeros_like(base_image), np.zeros_like(base_image), np.zeros_like(base_image),
                             np.zeros_like(base_image)])

    # Get all items on the canvas
    all_items = self.drg_segment_canvas.find_all()

    # Iterate through all items, deleting them except the image
    for item in all_items:
        if item != self.drg_segment_canvas_image_item:
            self.drg_segment_canvas.itemconfig(item, state='hidden')

    for seed, _ in self.seeds[self.current_image_index][:]:
        adjusted_projections = radial_projection_with_adjustment(self, edge_map, seed)
        roi_map = generate_roi_from_projections(edge_map, seed, adjusted_projections)

        self.regions[self.current_image_index].append(roi_map)
        composite_roi_map = np.logical_or(composite_roi_map, roi_map).astype(np.uint8)

        masked_region = np.where(roi_map != 0, base_image, 0)
        isodata_threshold = filters.threshold_isodata(masked_region[masked_region > 0])
        isodata_region = np.where(masked_region >= isodata_threshold, base_image, 0)
        self.positive_areas[self.current_image_index].append(np.count_nonzero(isodata_region))
        self.positive_intensities[self.current_image_index].append(np.mean(isodata_region[isodata_region > 0]))

    inverse_masked_region = np.where(composite_roi_map == 0, base_image, 0)

    li_threshold = filters.threshold_li(inverse_masked_region[inverse_masked_region > 0])
    li_region = np.where(inverse_masked_region >= li_threshold, base_image, 0)
    self.background_intensities[self.current_image_index] = np.mean(li_region[li_region != 0])

    for i in range(0, len(self.positive_areas[self.current_image_index])):
        intensity_diff = self.positive_intensities[self.current_image_index][i] - self.background_intensities[
            self.current_image_index]
        self.ctcf[self.current_image_index].append(
            intensity_diff * self.positive_areas[self.current_image_index][i])

    # Existing grow_regions code to calculate ROI areas and CTCF...
    for i, roi_area in enumerate(self.positive_areas[self.current_image_index]):
        ctcf_value = self.ctcf[self.current_image_index][i]
        # Update the corresponding table row with the calculated ROI area and CTCF value
        if i < len(self.table_item_ids):
            self.info_table.item(self.table_item_ids[i], values=(
                self.image_labels[self.current_image_index],  # Label
                f"{i + 1}",  # Seed#
                f"{self.seeds[self.current_image_index][i][0]}",  # Seed pos (assuming [i][0] is the seed position)
                f"{roi_area}",  # ROI area
                f"{ctcf_value}"  # CTCF
            ))
    self.save_data_button["state"] = "normal"

    roi_overlay[composite_roi_map == 1] = [255, 0, 0, 128]

    photo = df.convert_to_photoimage(roi_overlay)
    self.roi_photo = photo
    self.drg_segment_canvas_roi = self.drg_segment_canvas.create_image(0, 0, anchor='nw', image=photo)
    self.drg_segment_canvas.tag_raise(self.drg_segment_canvas_roi, self.drg_segment_canvas_image_item)


# Image Processing Utilities

def apply_contrast(self):
    """
    Apply contrast enhancement to the current image.
    """
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    gray = self.gray_images[self.current_image_index]
    # First, apply non-local means denoising
    # Since your images are not too large, starting parameters are chosen to be moderate.
    # These parameters can be adjusted based on the noise levels observed in your images.
    gray = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)

    enhanced = clahe.apply(gray)
    self.contrasted_gray_images[self.current_image_index] = enhanced
    single_channel_im = cv2.merge([np.zeros_like(enhanced), enhanced, np.zeros_like(enhanced)])
    contrasted_photo = df.convert_to_photoimage(single_channel_im)
    self.drg_segment_photo = contrasted_photo
    self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=contrasted_photo)


def radial_projection_with_adjustment(self, edge_map, seed, angle_step=1):
    """
    Adjust radial projections from a seed point to capture cell boundaries.
    """
    projections = []
    angles = np.arange(0, 360, angle_step)

    max_length = int(np.max(self.DRG_diameter[self.current_image_index]) * 1.2 / 2)
    # Calculate projections
    for angle in angles:
        for length in range(1, max_length + 1):
            dx = int(length * np.cos(np.radians(angle)))
            dy = int(length * np.sin(np.radians(angle)))
            x, y = seed[0] + dx, seed[1] + dy
            if not (0 <= x < edge_map.shape[1] and
                    0 <= y < edge_map.shape[0]) or edge_map[y, x] == 255 or length == max_length:
                projections.append(length)
                break

    for i in range(0, len(projections)):
        if projections[i] == max_length:
            projections[i] = np.median(projections)

    for i in range(0, len(projections)):
        if projections[i] == max_length:
            projections[i] = np.median(projections)

    return projections


def generate_roi_from_projections(edge_map, seed, adjusted_projections, angle_step=1):
    """
    Generate a Region Of Interest (ROI) from the adjusted projections.
    """
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


# User Interface Interaction Functions

def select_and_load_files(self):
    """
    Open a file dialog for the user to select image files, then load and process them.
    """
    file_types = [('Image files', '*.tiff;*.tif;*.jpg;*.jpeg;*.png;*.bmp'), ('All files', '*.*')]
    file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=file_types)
    if file_paths:
        self.image_file_paths = file_paths
        load_and_resize_images(self)
        df.prompt_for_labels(self)


def load_and_resize_images(self):
    """
    Load, resize, and process images selected by the user.
    """
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
        self.edge_thresholds.append([50, 150, 20])
        self.seeds.append([])
        self.process_statuses.append([False, False, False])
        self.regions.append([])
        self.positive_areas.append([])
        self.positive_intensities.append([])
        self.background_intensities.append(0)
        self.ctcf.append([])


def save_session_data_to_excel(self):
    # Initialize an empty list to hold the flat list of seed data
    seeds_data_flat_list = []

    # Iterate through the session data to extract only the seeds data
    for session_item in self.session_data:
        # Assuming 'seeds_data' key contains the seeds information for this session item
        seeds_data_list = session_item['seeds_data']

        # Further assuming seeds_data_list is a list of dictionaries
        for seed_data in seeds_data_list:
            # Optionally, you can modify or add additional keys here as needed
            seeds_data_flat_list.append(seed_data)

    # Get the current timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # Format as you like

    # Use the timestamp in the file name
    file_path = f'seeds_data_{timestamp}.xlsx'

    # Convert the flat list of seeds data into a pandas DataFrame
    df = pd.DataFrame(seeds_data_flat_list)

    # Save the DataFrame to an Excel file with the timestamped file name
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Seeds Data')

    messagebox.showinfo("Success", f"Seeds data saved successfully to {file_path}")


def save_current_image_data(self):
    """
    Saves the data from the table for the current image into the aggregated session data.
    """
    current_image_data = {
        "image_label": self.image_labels[self.current_image_index],
        "seeds_data": []
    }

    if not self.table_item_ids:
        messagebox.showerror("Error", "Table is empty", parent=self.drg_segment_window)
    else:
        # Retrieve data from the table
        for item_id in self.table_item_ids:
            item = self.info_table.item(item_id)
            values = item["values"]
            seed_data = {
                "label": values[0],
                "seed_number": values[1],
                "seed_position": values[2],
                "roi_area": values[3],
                "ctcf": values[4]
            }
            current_image_data["seeds_data"].append(seed_data)

        # Check if we are updating an existing entry or adding a new one
        if len(self.session_data) > self.current_image_index:
            self.session_data[self.current_image_index] = current_image_data
        else:
            self.session_data.append(current_image_data)

        # Close the DRG segmentation window
        self.drg_segment_window.destroy()

        self.table_item_ids.clear()
        # Optional: Show a confirmation message
        messagebox.showinfo("Data Saved", "The segmentation data has been saved successfully.")


# Image Transformation Functions

def resize_image(self, image):
    """
    Resize an image to fit within specified dimensions while maintaining aspect ratio.
    """
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


def draw_diameters(self):
    # Create a new window for DRG Segmentation
    self.seeds_button["state"] = "disabled"
    self.edges_button["state"] = "disabled"
    self.regions_button["state"] = "disabled"
    messagebox.showinfo("Instructions", "Left-click to start and end a line. Right-click to remove a line.",
                        parent=self.drg_segment_window)
    if hasattr(self, 'drg_segment_canvas_roi'):
        self.drg_segment_canvas.delete(self.drg_segment_canvas_roi)

    # Get all items on the canvas
    all_items = self.drg_segment_canvas.find_all()

    # Iterate through all items, deleting them except the image
    for item in all_items:
        if item != self.drg_segment_canvas_image_item:
            self.drg_segment_canvas.itemconfig(item, state='normal')

    def save_diameters():
        if not self.DRG_line_ids:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires drawing at least 1 cell diameter",
                                 parent=self.drg_segment_window)

        else:
            self.DRG_diameter[self.current_image_index] = [np.linalg.norm(np.array(start) - np.array(end)) for
                                                           start, end, _ in self.DRG_line_ids]
            # Clear lines information
            self.DRG_line_ids.clear()
            messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
            self.process_statuses[self.current_image_index][2] = True

            self.drg_segment_canvas.unbind("<Button-1>")
            self.drg_segment_canvas.unbind("<B1-Motion>")
            self.drg_segment_canvas.unbind("<ButtonRelease-1>")
            self.drg_segment_canvas.unbind("<Button-3>")  # Right-click to delete a line

            self.seeds_button["state"] = "normal"
            self.edges_button["state"] = "normal"
            self.regions_button["state"] = "normal"
            save_button.destroy()

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
                                                                 fill="#ff0000", width=3, tags="temp_line")

    def end_line(event):
        """Function to finalize drawing a line."""
        if self.draw_start:
            line_id = self.drg_segment_canvas.create_line(self.draw_start[0], self.draw_start[1], event.x, event.y,
                                                          fill="#ff0000", width=3)
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

    save_button = tk.Button(self.drg_segment_window, text="Save", command=lambda: save_diameters())
    save_button.place(anchor="nw", x=self.lines_button.winfo_x() + self.lines_button.winfo_width() + 10,
                      y=self.lines_button.winfo_y())

    # Bind events for line drawing
    self.drg_segment_canvas.bind("<Button-1>", start_line)
    self.drg_segment_canvas.bind("<B1-Motion>", draw_line)
    self.drg_segment_canvas.bind("<ButtonRelease-1>", end_line)
    self.drg_segment_canvas.bind("<Button-3>", delete_line)  # Right-click to delete a line

    self.drg_segment_window.update_idletasks()


def set_seeds(self):
    # Create a new window for DRG Segmentation
    self.lines_button["state"] = "disabled"
    self.edges_button["state"] = "disabled"
    self.regions_button["state"] = "disabled"
    messagebox.showinfo("Instructions",
                        "Left click to add seeds (center of each cell to count). \nRight click to remove",
                        parent=self.drg_segment_window)

    if self.table_item_ids:
        for id in self.table_item_ids:
            self.info_table.delete(id)

    if self.seeds[self.current_image_index]:
        # Existing code to check seeds and clear the table
        self.table_item_ids.clear()  # Clear existing references
        for i, (coords, seed_id) in enumerate(self.seeds[self.current_image_index]):
            row_id = self.info_table.insert("", tk.END, values=(
                self.image_labels[self.current_image_index], f"{i + 1}", f"{coords}", "n/a",
                "n/a"))
            self.table_item_ids.append(row_id)

    if hasattr(self, 'drg_segment_canvas_roi'):
        self.drg_segment_canvas.delete(self.drg_segment_canvas_roi)

    # Get all items on the canvas
    all_items = self.drg_segment_canvas.find_all()

    # Iterate through all items, deleting them except the image
    for item in all_items:
        if item != self.drg_segment_canvas_image_item:
            self.drg_segment_canvas.itemconfig(item, state='normal')

    def save_seeds():
        if not self.seeds[self.current_image_index]:
            # Display instruction message box
            messagebox.showerror("Error", "Segmentation requires at least 1 seed", parent=self.drg_segment_window)

        else:
            if self.table_item_ids:
                for id in self.table_item_ids:
                    self.info_table.delete(id)
            # Existing code to check seeds and clear the table
            self.table_item_ids.clear()  # Clear existing references
            for i, (coords, seed_id) in enumerate(self.seeds[self.current_image_index]):
                row_id = self.info_table.insert("", tk.END, values=(
                    self.image_labels[self.current_image_index], f"{i + 1}", f"{coords}", "n/a",
                    "n/a"))
                self.table_item_ids.append(row_id)

            self.lines_button["state"] = "normal"
            self.edges_button["state"] = "normal"
            self.regions_button["state"] = "normal"
            save_button.destroy()
            messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
            self.process_statuses[self.current_image_index][1] = True
            self.drg_segment_canvas.unbind("<Button-1>")
            self.drg_segment_canvas.unbind("<Button-3>")

    def add_seed(event):
        seed_id = self.drg_segment_canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                                                      fill='#ff0000', outline='#ff0000', width=2)
        # Convert coords to tuple before adding
        self.seeds[self.current_image_index].append((tuple(np.array((event.x, event.y))), seed_id))

    def delete_seed(event):
        proximity_threshold = 20 * self.root.winfo_screenwidth() / 2560  # Adjust threshold as necessary
        event_coords = np.array((event.x, event.y))
        for i, (coords, seed_id) in enumerate(list(self.seeds[self.current_image_index])):
            if np.linalg.norm(coords - event_coords) <= proximity_threshold:
                self.drg_segment_canvas.delete(seed_id)
                # Remove by index to avoid ambiguity
                del self.seeds[self.current_image_index][i]
                break

    save_button = tk.Button(self.drg_segment_window, text="Save", command=lambda: save_seeds())
    save_button.place(anchor="nw", x=self.seeds_button.winfo_x() + self.seeds_button.winfo_width() + 10,
                      y=self.seeds_button.winfo_y())

    self.drg_segment_canvas.bind("<Button-1>", add_seed)
    self.drg_segment_canvas.bind("<Button-3>", delete_seed)

    self.drg_segment_window.update_idletasks()


def edge_detect(self):
    # Create a new window for DRG Segmentation
    edges_window = tk.Toplevel(self.root)
    edges_window.attributes('-topmost', True)
    edges_window.title("Edge Detection Settings")

    initial_image = self.contrasted_gray_images[self.current_image_index]
    single_channel_im = cv2.merge([np.zeros_like(initial_image), initial_image, np.zeros_like(initial_image)])
    photo = df.convert_to_photoimage(single_channel_im)
    self.drg_segment_image = single_channel_im
    self.drg_segment_photo = single_channel_im
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
            edge_overlay = single_channel_im.copy()
            edge_overlay[filtered_map == 255] = [255, 255, 255]

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

            # If the nearest contour is found and within deletion threshold, delete it
            if nearest_contour_index is not None:
                cv2.drawContours(self.edge_maps[self.current_image_index], [contours[nearest_contour_index]], -1, 0,
                                 thickness=-1)

            # Refresh the display after contour deletion
            # Update the display with filtered edge map
            edge_overlay = single_channel_im.copy()
            edge_overlay[self.edge_maps[self.current_image_index] == 255] = [255, 255, 255]
            edge_photo = df.convert_to_photoimage(edge_overlay)
            self.drg_segment_photo = edge_photo
            self.drg_segment_canvas.itemconfig(self.drg_segment_canvas_image_item, image=edge_photo)

    def save_edges():
        self.edge_thresholds[self.current_image_index][0] = t1_slider_value.get()
        self.edge_thresholds[self.current_image_index][1] = t2_slider_value.get()
        self.edge_thresholds[self.current_image_index][2] = min_slider_value.get()

        edges_window.destroy()
        messagebox.showinfo("Status", "Save successful!", parent=self.drg_segment_window)
        self.process_statuses[self.current_image_index][0] = True
        self.drg_segment_canvas.unbind("<Button-3>")

    apply_edges(self.edge_thresholds[self.current_image_index][0], self.edge_thresholds[self.current_image_index][1],
                self.edge_thresholds[self.current_image_index][2], initial_image)

    t1_slider_value = tk.IntVar(value=self.edge_thresholds[self.current_image_index][0])
    t2_slider_value = tk.IntVar(value=self.edge_thresholds[self.current_image_index][1])
    min_slider_value = tk.IntVar(value=self.edge_thresholds[self.current_image_index][2])

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
