import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

##############################################################################
# 1. Parameters for SDF & Random-Center Circles (Same as Your Correct Code)
##############################################################################

width, height = 1000, 1000   # Same dimensions as your original circle code
margin = 150                # For random center
radii = [100, 80, 60]
noise_scale = 0.02
noise_intensity = 5
sprite_count = 15

def get_random_center(width, height, margin):
    """Random center within [margin, width-margin] x [margin, height-margin]."""
    center_x = np.random.randint(margin, width - margin)
    center_y = np.random.randint(margin, height - margin)
    return center_x, center_y

def generate_simple_noise(width, height, scale=0.05, intensity=10):
    """Generate sine-cosine-based noise."""
    y, x = np.meshgrid(np.arange(height), np.arange(width))
    noise = np.sin(x * scale) + np.cos(y * scale)
    return noise * intensity

def create_sprite(center_x, center_y, radii, noise_scale=0.05, noise_intensity=10):
    """
    Create a layered SDF (1000x1000) with noise for the given center and radii.
    """
    y, x = np.meshgrid(np.arange(height), np.arange(width))
    combined_sdf = np.zeros((height, width))

    for radius in radii:
        sdf = np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius
        noise = generate_simple_noise(width, height, scale=noise_scale, intensity=noise_intensity)
        sdf_with_noise = sdf + noise
        combined_sdf = np.minimum(combined_sdf, sdf_with_noise)

    return combined_sdf

##############################################################################
# 2. Set Up PyQtGraph Window (White Background)
##############################################################################

app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(show=True, title="Random-Center SDF Circles (Ring Spinner)")
win.resize(800, 800)

plot = win.addPlot()
plot.setAspectLocked(True)
plot.hideAxis('bottom')
plot.hideAxis('left')

# Make background white (so circles appear as in Matplotlib)
win.setBackground('w')

##############################################################################
# 3. Generate Circles Using Your Original Logic
##############################################################################

# Cycle through colormaps as you do in Matplotlib
sequential_colormaps = [
    plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds,
    plt.cm.YlOrBr, plt.cm.YlOrRd, plt.cm.OrRd, plt.cm.PuRd, plt.cm.RdPu, plt.cm.BuPu,
    plt.cm.GnBu, plt.cm.PuBu, plt.cm.YlGnBu, plt.cm.PuBuGn, plt.cm.BuGn, plt.cm.YlGn
]
colormap_cycle = cycle(sequential_colormaps)

sprites = []
for _ in range(sprite_count):
    # Random center => unique circle shape
    cx, cy = get_random_center(width, height, margin)
    layered_sdf = create_sprite(cx, cy, radii, noise_scale, noise_intensity)
    
    # Normalize SDF to [0,1]
    normalized_sdf = (layered_sdf - layered_sdf.min()) / (layered_sdf.max() - layered_sdf.min())
    
    # Apply colormap
    cmap = next(colormap_cycle)
    colors_layered = cmap(normalized_sdf)  # shape: (1000, 1000, 4), floats in [0,1]
    
    # Alpha channel
    alpha_layered = np.clip(1 - normalized_sdf**2, 0, 1)
    colors_layered[..., -1] = alpha_layered
    
    # Convert float [0..1] => uint8 [0..255]
    sprite_rgba_8u = (colors_layered * 255).astype(np.uint8)
    sprites.append(sprite_rgba_8u)

# Create PyQtGraph ImageItems for each circle
sprite_items = []
for s in sprites:
    item = pg.ImageItem(image=s)
    # For correct row-major alignment
    item.setOpts(axisOrder='row-major')
    plot.addItem(item)
    sprite_items.append(item)

##############################################################################
# 4. Arrange in a Ring and Animate
##############################################################################

# Arrange sprites evenly spaced in a ring
ring_radius = 300
angles = np.linspace(0, 2 * np.pi, sprite_count, endpoint=False)

# Place the circles initially in the correct positions
for i, item in enumerate(sprite_items):
    theta = angles[i]
    x = ring_radius * np.cos(theta)
    y = ring_radius * np.sin(theta)
    item.setPos(x - width / 2, y - height / 2)  # Subtract half the sprite size for centering

# Rotate them in the ring during the animation
def update():
    global angles
    rotation_speed = 0.01  # Adjust for faster/slower rotation
    angles += rotation_speed
    
    for i, item in enumerate(sprite_items):
        theta = angles[i]
        x = ring_radius * np.cos(theta)
        y = ring_radius * np.sin(theta)
        item.setPos(x - width / 2, y - height / 2)

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)  # ~60 FPS

QtWidgets.QApplication.instance().exec_()