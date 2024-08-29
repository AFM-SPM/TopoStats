import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageColor
import numpy as np
import os

# Predefined colors with numbers assigned, change this if more classes are required
PREDEFINED_COLORS = [
    (1, "Red", "#ff0000"),
    (2, "Blue", "#0000ff"),
    (3, "Green", "#00ff00"),
    (4, "Yellow", "#ffff00"),
    (5, "Purple", "#800080")
]

# Create a dictionary to map color hex codes to their numbers
COLOR_TO_NUMBER = {hex_color: number for number, name, hex_color in PREDEFINED_COLORS}
NUMBER_TO_COLOR = {number: hex_color for number, name, hex_color in PREDEFINED_COLORS}

class TopoLabel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TopoLabelling")

        self.image = None
        self.mask = None
        self.draw = None
        self.stroke_stack = []
        self.erased_strokes = []
        self.brush_color = "#ffffff"
        self.brush_size = 5  # Initialize brush size
        self.brush_number = 1
        self.last_x, self.last_y = None, None
        self.has_strokes = False
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.image_files = []
        self.current_index = -1
        self.current_folder = ""
        self.is_eraser = False

        self.setup_ui()

    def setup_ui(self):
        # Create a frame to hold the canvas and the scrollbars
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create the canvas with scrollbars
        self.canvas = tk.Canvas(frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar configuration
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar configuration
        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Set scrollbars to the canvas
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        # Create UI elements
        self.open_button = tk.Button(self.root, text="Open Folder", command=self.open_folder)
        self.open_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.root, text="Save Current Mask", command=self.save_brush_strokes)
        self.save_button.pack(side=tk.RIGHT)

        self.erase_button = tk.Button(self.root, text="Erase", command=self.toggle_eraser)
        self.erase_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(self.root, text="Previous", command=self.previous_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT)

        self.selected_color = tk.StringVar(self.root)
        self.selected_color.set(PREDEFINED_COLORS[0][1])
        self.color_dropdown = tk.OptionMenu(
            self.root,
            self.selected_color,
            *[name for number, name, hex_color in PREDEFINED_COLORS],
            command=self.color_selected
        )
        self.color_dropdown.pack(side=tk.LEFT)

        self.color_display = tk.Label(self.root, text="Brush Color")
        self.color_display.pack(side=tk.LEFT)

        self.brush_size_label = tk.Label(self.root, text=f"Brush Size: {self.brush_size}")
        self.brush_size_label.pack(side=tk.LEFT)

        self.brush_size_slider = tk.Scale(self.root, from_=1, to=50, orient=tk.HORIZONTAL, command=self.update_brush_size)
        self.brush_size_slider.set(self.brush_size)
        self.brush_size_slider.pack(side=tk.LEFT)

        self.zoom_slider = tk.Scale(self.root, from_=self.min_zoom, to=self.max_zoom, resolution=0.1, orient=tk.HORIZONTAL, label="Zoom", command=self.update_zoom)
        self.zoom_slider.set(self.zoom_level)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.root.bind("<ButtonRelease-1>", self.reset_last_coords)
        self.root.bind("<space>", self.next_image)

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        
        if folder_path:
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if self.image_files:
                self.current_index = 0
                self.current_folder = folder_path
                self.load_image(self.image_files[self.current_index], folder_path)
                self.update_navigation_buttons()
            else:
                messagebox.showerror("Error", "No image files found in the selected folder.")

    def load_image(self, image_file, folder_path):
        image_path = os.path.join(folder_path, image_file)
        try:
            self.image = Image.open(image_path).convert("RGBA")
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {image_path}")
            return
        
        self.mask = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.mask)
        self.stroke_stack.clear()
        self.erased_strokes.clear()

        self.display_image()

    def display_image(self):
        if self.image is not None:
            # Resize the image according to the zoom level
            zoomed_image = self.image.resize(
                (int(self.image.width * self.zoom_level), int(self.image.height * self.zoom_level)),
                Image.Resampling.LANCZOS
            )
            zoomed_mask = self.mask.resize(
                (int(self.mask.width * self.zoom_level), int(self.mask.height * self.zoom_level)),
                Image.Resampling.NEAREST
            )

            # Combine the image and mask for display
            combined_image = Image.alpha_composite(zoomed_image, zoomed_mask)
            photo = ImageTk.PhotoImage(combined_image)

            # Set the canvas scroll region to match the zoomed image size
            self.canvas.config(scrollregion=(0, 0, combined_image.width, combined_image.height))

            # Update the canvas image
            self.canvas.delete(tk.ALL)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo

    def reset_last_coords(self, event):
        self.last_x, self.last_y = None, None

    def save_brush_strokes(self):
        if self.mask is None:
            messagebox.showerror("Error", "No brush strokes to save! Please load an image and draw first.")
            return False  # Return False since there's nothing to save

        if self.image_files:
            current_image_file = self.image_files[self.current_index]
            base_name, ext = os.path.splitext(current_image_file)
            default_save_path_npy = os.path.join(self.current_folder, f"{base_name}_mask.npy")
            default_save_path_png = os.path.join(self.current_folder, f"{base_name}_mask.png")
        else:
            default_save_path_npy = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
            default_save_path_png = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        
        if not default_save_path_npy or not default_save_path_png:
            return False

        mask_array = np.array(self.mask)
        num_array = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        
        for number, _, hex_color in PREDEFINED_COLORS:
            r, g, b = ImageColor.getrgb(hex_color)
            color_mask = (mask_array[..., :3] == [r, g, b]).all(axis=-1)
            num_array[color_mask] = number
        
        background_mask = (mask_array[..., 3] == 0)
        num_array[background_mask] = 0
        
        try:
            # Save as .npy
            np.save(default_save_path_npy, num_array)
            
            # Create a black background image and paste the mask on top
            black_background = Image.new("RGB", self.mask.size, (0, 0, 0))
            rgb_mask = Image.alpha_composite(black_background.convert("RGBA"), self.mask).convert("RGB")
            
            # Save as .png with black background
            rgb_mask.save(default_save_path_png)
            
            messagebox.showinfo("Success", f"Brush strokes saved as: {default_save_path_npy} and {default_save_path_png}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save the file: {e}")
            return False

    def paint(self, event):
        x, y = event.x, event.y
        
        if self.last_x is not None and self.last_y is not None:
            scaled_x1 = int(self.last_x / self.zoom_level)
            scaled_y1 = int(self.last_y / self.zoom_level)
            scaled_x2 = int(x / self.zoom_level)
            scaled_y2 = int(y / self.zoom_level)

            if self.is_eraser:
                self.draw.line([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill=(0, 0, 0, 0), width=self.brush_size * 2)
                self.draw.ellipse([scaled_x2 - self.brush_size, scaled_y2 - self.brush_size, scaled_x2 + self.brush_size, scaled_y2 + self.brush_size], fill=(0, 0, 0, 0), outline=(0, 0, 0, 0))
            else:
                self.draw.line([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill=self.brush_color, width=self.brush_size * 2)
                self.draw.ellipse([scaled_x2 - self.brush_size, scaled_y2 - self.brush_size, scaled_x2 + self.brush_size, scaled_y2 + self.brush_size], fill=self.brush_color, outline=self.brush_color)

            self.stroke_stack.append((self.brush_color, scaled_x1, scaled_y1, scaled_x2, scaled_y2, self.brush_size))
            self.has_strokes = True

        self.last_x, self.last_y = x, y
        
        self.display_image()

    def color_selected(self, color_name):
        # Find the number and hex color code for the selected colour name
        for number, name, hex_color in PREDEFINED_COLORS:
            if name == color_name:
                self.brush_number = number
                self.brush_color = hex_color
                break
        self.color_display.config(bg=self.brush_color)
        self.is_eraser = False

    def update_brush_size(self, val):
        self.brush_size = int(val)
        self.brush_size_label.config(text=f"Brush Size: {self.brush_size}")

    def toggle_eraser(self):
        self.is_eraser = not self.is_eraser
        if self.is_eraser:
            self.erase_button.config(relief=tk.SUNKEN)
        else:
            self.erase_button.config(relief=tk.RAISED)

    def update_zoom(self, val):
        self.zoom_level = float(val)
        self.display_image()

    def next_image(self, event=None):
        if self.has_strokes:
            if not self.save_brush_strokes():
                return  # If saving fails or is cancelled, do not move to the next image
        
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image(self.image_files[self.current_index], self.current_folder)
            self.update_navigation_buttons()
            self.has_strokes = False  # Reset the flag after moving to the next image

    def previous_image(self, event=None):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.load_image(self.image_files[self.current_index], self.current_folder)
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.image_files) - 1 else tk.DISABLED)

# Create the TopoLabel instance and start the main loop
root = TopoLabel().root
root.mainloop()