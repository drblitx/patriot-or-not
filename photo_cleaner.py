"""
@name photo_cleaner.py
@author drblitx
@created July 2025
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button

input_folder = 'data/other'
output_folder = 'data/final-other/'
os.makedirs(output_folder, exist_ok=True)

crop_coords = []
confirm_crop = None

def onselect(eclick, erelease):
    global crop_coords
    crop_coords = [int(eclick.xdata), int(eclick.ydata),
                   int(erelease.xdata), int(erelease.ydata)]

def confirm(event):
    global confirm_crop
    confirm_crop = True
    plt.close()

def skip(event):
    global confirm_crop
    confirm_crop = False
    plt.close()

# process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        continue

    img_path = os.path.join(input_folder, filename)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # ensure compatibility
            crop_coords = []
            confirm_crop = None

            # Step 1: crop selector
            fig1, ax1 = plt.subplots()
            ax1.imshow(img)
            ax1.set_title(f"Draw crop: {filename}")
            RectangleSelector(
                ax1, onselect, useblit=True,
                button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=True
            )
            plt.show()
            plt.close(fig1)

            # Step 2: crop preview and confirm
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                cropped = img.crop((x1, y1, x2, y2)).resize((224, 224))

                fig2, ax2 = plt.subplots()
                plt.subplots_adjust(bottom=0.2)
                ax2.imshow(cropped)
                ax2.set_title("Cropped Result")
                ax2.axis('off')

                ax_confirm = plt.axes([0.25, 0.05, 0.2, 0.075])
                ax_skip = plt.axes([0.55, 0.05, 0.2, 0.075])
                Button(ax_confirm, '✅ Confirm').on_clicked(confirm)
                Button(ax_skip, '❌ Skip').on_clicked(skip)

                plt.show()
                plt.close(fig2)

                if confirm_crop:
                    cropped.save(os.path.join(output_folder, filename))
                    print(f"✅ Saved CROP: {filename}")
                    continue  # move to next image

            # Fallback: save resized original
            resized = img.resize((224, 224))
            resized.save(os.path.join(output_folder, filename))
            print(f"📎 Saved ORIGINAL: {filename}")

    except Exception as e:
        print(f"⚠️ Error with {filename}: {e}")
