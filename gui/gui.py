#!/pkg/mamba/envs/yolo8/bin/python

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import subprocess
from tkinter import ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk

selected_path = ""  # Initialize the selected path variable

def select_folder():
    global selected_path, selected_label, selection_type_label
    selected_path = filedialog.askdirectory()
    if selected_path:
        selected_label.config(text="Selected Path: \n" + selected_path)
        selection_type_label.config(text="You selected folder:")
    else:
        selected_label.config(text="No folder selected")

def select_image():
    global selected_path, selected_label, selection_type_label
    selected_path = filedialog.askopenfilename()
    if selected_path:
        selected_label.config(text="Selected Path: \n" + selected_path)
        selection_type_label.config(text="You selected image:")
    else:
        selected_label.config(text="No image selected")

def set_threshold_egg(value):
    global threshold_egg
    threshold_egg = round(float(value), 3)
    threshold_label_1.config(text=f"Threshold EGG: {threshold_egg}")

def set_threshold(value):
    global threshold
    threshold = round(float(value), 3)
    threshold_label.config(text=f"Threshold JUV: {threshold}")

def set_grid_value(*args):
    global grid_value
    grid_value = grid_var.get()
    grid_label.config(text=f"Grid Value: {grid_value}")

def select_output_csv():
    global output_csv_path
    output_csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if output_csv_path:
        output_csv_label.config(text="Output CSV: \n" + output_csv_path)

def select_output_images():
    global output_images_path
    output_images_path = filedialog.askdirectory()
    if output_images_path:
        output_images_label.config(text="Output Images Folder: \n" + output_images_path)


# Initialize global variables
threshold = 0.5
threshold_egg = 0.5
grid_value = ""
output_csv_path = ""
output_images_path = ""

# Create main window
root = tk.Tk()
root.title("SEGNEMA")

# Load and display logo
logo_path = "/doc/code/DEG/nematodes/other/testing/nema.png"  # Replace with the actual path to your logo
logo = Image.open(logo_path)

# Scale down by a factor of three
new_width = logo.width // 2
new_height = logo.height // 2
logo = logo.resize((new_width, new_height))

logo = ImageTk.PhotoImage(logo)
logo_label = ttk.Label(root, image=logo)
logo_label.pack(pady=10)

# Create buttons and widgets
select_folder_button = ttk.Button(root, text="Select Folder", command=select_folder)
select_folder_button.pack(pady=5)

selected_label = ttk.Label(root, text="or")
selected_label.pack()

select_image_button = ttk.Button(root, text="Select Image", command=select_image)
select_image_button.pack(pady=5)

selected_label = ttk.Label(root, text="Selected Path:")
selected_label.pack()

selection_type_label = ttk.Label(root, text="")
selection_type_label.pack()

separator1 = ttk.Separator(root, orient="horizontal")
separator1.pack(fill="x", padx=5, pady=5)

threshold_slider = ttk.Scale(root, from_=0, to=1, orient="horizontal", value=0.7, command=set_threshold)
threshold_slider.pack(pady=10)
threshold_label = ttk.Label(root, text="Threshold JUV: 0.5")
threshold_label.pack()

threshold_slider_1 = ttk.Scale(root, from_=0, to=1, orient="horizontal", value=0.7, command=set_threshold_egg)
threshold_slider_1.pack(pady=10)
threshold_label_1 = ttk.Label(root, text="Threshold EGG: 0.5")
threshold_label_1.pack()

separator2 = ttk.Separator(root, orient="horizontal")
separator2.pack(fill="x", padx=5, pady=5)

grid_entry_label = ttk.Label(root, text="Set Grid:")
grid_entry_label.pack()

grid_var = tk.StringVar()
grid_var.trace("w", set_grid_value)

grid_entry = ttk.Entry(root, textvariable=grid_var)
grid_entry.pack()

grid_label = ttk.Label(root, text="Grid Value:")
grid_label.pack()

separator3 = ttk.Separator(root, orient="horizontal")
separator3.pack(fill="x", padx=5, pady=5)

output_csv_button = ttk.Button(root, text="Select Output CSV Path", command=select_output_csv)
output_csv_button.pack(pady=10)
output_csv_label = ttk.Label(root, text="Output CSV:")
output_csv_label.pack()

output_images_button = ttk.Button(root, text="Select Output Images Folder", command=select_output_images)
output_images_button.pack(pady=10)
output_images_label = ttk.Label(root, text="Output Images Folder:")
output_images_label.pack()

separator4 = ttk.Separator(root, orient="horizontal")
separator4.pack(fill="x", padx=5, pady=5)

def run_script():
    global selected_path, threshold, grid_value, output_csv_path, output_images_path
    if not selected_path:
        messagebox.showerror("Error", "Please select a folder or an image.")
        return

    command = [
        '/pkg/mamba/envs/yolo8/bin/python', 
        '/doc/code/DEG/nematodes/other/testing/nema.py', 
        selected_path,
        "--thresh_juv", str(threshold),
        "--thresh_egg", str(threshold_egg),
        "--grid_size", grid_value,
        "--output_csv", output_csv_path,
        "--save_folder", output_images_path
    ]
    subprocess.run(command)

run_button = ttk.Button(root, text="   Run Script   ", command=run_script, style="Run.TButton")
run_button.pack(pady=20)

# Add a custom style for the Run button
style = ttk.Style()
style.configure("Run.TButton", background="orange", foreground="black", font=("Helvetica", 16, "bold"))

# Start the main event loop
root.mainloop()
