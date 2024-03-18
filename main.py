import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
import cv2
import cv_functions as cvf
import display_functions as df


class ImageLabelerApp:
    def __init__(self, root):
        """
        Initialize the Image Labeler application with UI elements and variables.
        """
        # Set up the main window
        self.root = root
        self.root.state('zoomed')  # Maximize the window
        self.root.title("DRGvisual")
        # Canvas for image display
        self.setup_canvas()

        # Image and processing related variables
        self.setup_image_variables()

        # UI components like buttons and labels
        self.setup_ui_components()

        # Event handlers for interactions
        self.setup_event_handlers()

        # Finalize UI setup
        self.root.after(100, self.finalize_ui_setup)

    def setup_canvas(self):
        """Set up the canvas for image display."""
        self.canvas = tk.Canvas(self.root, bg='gray')
        self.canvas.pack(fill='both', expand=True)

    def setup_image_variables(self):
        """Initialize variables related to image processing."""
        # Arrays for storing image data
        self.cv2_images = []
        self.photoimages = []
        self.gray_images = []
        self.contrasted_gray_images = []
        self.edge_maps = []
        self.session_data = []  # To hold data for all images
        # Assuming self.table_item_ids stores references to each row inserted into the table
        self.table_item_ids = []

        # Variables for image processing states and parameters
        self.edge_thresholds = []
        self.seeds = []
        self.process_statuses = []
        self.regions = []
        self.positive_areas = []
        self.positive_intensities = []
        self.background_intensities = []
        self.ctcf = []
        self.diameter_items = []  # Store diameter canvas item references
        self.region_items = []  # Store region canvas item references

        # Image file management
        self.image_file_paths = []
        self.image_labels = []
        self.DRG_diameter = []
        self.DRG_line_ids = []

        # Currently displayed image index
        self.current_image_index = None

    def setup_ui_components(self):
        """Initialize UI components such as buttons and labels."""
        # Buttons for navigation and actions
        self.prev_button = tk.Button(self.root, text="< Prev", command=lambda: df.prev_image(self))
        self.next_button = tk.Button(self.root, text="Next >", command=lambda: df.next_image(self))
        self.select_button = tk.Button(self.root, text="Select Image Files",
                                       command=lambda: cvf.select_and_load_files(self))
        self.rgb_split_button = tk.Button(self.root, text="RGB Preview", command=lambda: df.rgb_split(self))
        self.label_reset_button = tk.Button(self.root, text="Labels", command=lambda: df.prompt_for_labels(self))
        self.save_all_button = tk.Button(self.root, text="Save all", command=lambda: cvf.save_session_data_to_excel(self))
        # Dropdown for color channel selection
        self.setup_color_channel_dropdown()

        # DRG segmentation button
        self.drg_segment_button = tk.Button(self.root, text="DRG Segmentation", command=lambda: cvf.drg_segment(self))

    def setup_color_channel_dropdown(self):
        """Set up the dropdown menu for selecting color channels."""
        self.channel_var = tk.StringVar(self.root)
        self.color_channel_combobox = ttk.Combobox(self.root, textvariable=self.channel_var, state="readonly")
        self.color_channel_combobox['values'] = ('Red', 'Green', 'Blue')
        self.color_channel_combobox.current(1)  # Default selection

        # Label for the dropdown
        channel_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.channel_label = tk.Label(self.root, font=channel_font, bg='gray', text="DRG Channel")

    def setup_event_handlers(self):
        """Configure event handlers for mouse actions and window interactions."""
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel_scroll)

    def finalize_ui_setup(self):
        """Perform final steps in setting up the UI, including drawing the initial image rectangle."""
        self.rect_start_x = int(0.1 * self.root.winfo_screenwidth())
        self.rect_start_y = int(0.1 * self.root.winfo_screenheight()) - self.root.winfo_rooty()
        self.rect_end_x = int(0.9 * self.root.winfo_screenwidth())
        self.rect_end_y = int(0.9 * self.root.winfo_screenheight()) - self.root.winfo_rooty()
        self.rect_width = self.rect_end_x - self.rect_start_x
        self.rect_height = self.rect_end_y - self.rect_start_y
        self.rect_center_x = int(self.rect_start_x + self.rect_width / 2)
        self.rect_center_y = int(self.rect_start_y + self.rect_height / 2)

        self.canvas.create_rectangle(self.rect_start_x, self.rect_start_y,
                                     self.rect_end_x, self.rect_end_y,
                                     fill="black", outline="black")

        # Setup additional UI elements like labels and place buttons
        self.setup_labels_and_place_buttons()

    def setup_labels_and_place_buttons(self):
        """Create and place UI labels and buttons after canvas setup."""
        # Defining dimensions and positions for UI elements based on window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        button_width = int(0.07 * screen_width)
        button_height = int(0.025 * screen_height)  # Static height for simplicity
        wgap = int(0.008 * screen_width)  # Gap between buttons and other elements
        hgap = int(0.01 * screen_height)  # Gap between buttons and other elements

        # Calculate positions based on the defined dimensions and positions
        prev_button_x = self.rect_start_x
        next_button_x = prev_button_x + button_width + wgap
        select_button_x = self.rect_end_x + wgap
        buttons_y = self.rect_end_y + hgap

        # Place navigation buttons
        self.prev_button.place(x=prev_button_x, y=buttons_y, width=button_width, height=button_height)
        self.next_button.place(x=next_button_x, y=buttons_y, width=button_width, height=button_height)
        self.label_reset_button.place(x=prev_button_x, y=buttons_y+button_height+hgap, width=button_width, height=button_height)

        # Select image files button
        self.select_button.place(x=select_button_x, y=self.rect_start_y, height=button_height, width=button_width)

        # RGB split button positioned below the select button
        self.rgb_split_button.place(x=select_button_x, y=self.rect_start_y + button_height + hgap,
                                    height=button_height, width=button_width)

        # Place the channel label and dropdown for color channel selection
        channel_label_x = select_button_x
        channel_label_y = self.rect_start_y + 3 * (button_height + hgap)
        self.channel_label.place(x=channel_label_x, y=channel_label_y, height=button_height, width=button_width)

        # Dropdown below the channel label
        self.color_channel_combobox.place(x=channel_label_x, y=channel_label_y + button_height + hgap,
                                          height=button_height, width=button_width)
        # DRG Segmentation button below RGB split button
        self.drg_segment_button.place(x=channel_label_x,
                                      y=channel_label_y + 2*(button_height + hgap), height=button_height,
                                      width=button_width)
        self.save_all_button.place(x=channel_label_x, y=channel_label_y + 3*(button_height + hgap), height=button_height,
                                   width=button_width)
        # Image label display setup
        label_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.image_label_display = tk.Label(self.root, font=label_font, bg='white')
        self.image_label_display.config(text="")
        # Positioning the label just below the canvas area
        self.image_label_display.place(x=self.rect_center_x, y=self.rect_end_y+1, anchor="n")

        self.pan_start_x = None
        self.pan_start_y = None
        self.image_offset_x = 0
        self.image_offset_y = 0

    # Event handlers and other methods remain mostly unchanged, with added comments for clarity
    def on_mouse_press(self, event):
        """Handle mouse press event for starting pan operations."""
        # Initialize pan start coordinates
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_mouse_drag(self, event):
        """Handle mouse drag event for panning the image."""
        # Calculate delta and update image offset for panning
        if self.pan_start_x is not None and self.pan_start_y is not None:
            delta_x = event.x - self.pan_start_x
            delta_y = event.y - self.pan_start_y
            self.image_offset_x += delta_x
            self.image_offset_y += delta_y
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            df.display_current_image(self)  # Update display

    def on_mouse_wheel_scroll(self, event):
        """Zoom in or out based on mouse wheel movement."""
        # Retrieve the current image
        image = self.cv2_images[self.current_image_index]

        # Determine the zoom factor (10% zoom in or zoom out)
        zoom_factor = 1.1 if event.delta > 0 else 0.9

        # Calculate the new dimensions
        new_width = int(image.shape[1] * zoom_factor)
        new_height = int(image.shape[0] * zoom_factor)

        # Apply the resizing
        zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Update the image in the list
        self.cv2_images[self.current_image_index] = zoomed_image

        # Redraw the current image
        df.display_current_image(self)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelerApp(root)
    root.mainloop()
