import bpy
import bmesh
import os
import csv
import random
import bpy
import random
import os
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import bpy
import random
import os
import numpy as np
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import bpy
import random
import os
import numpy as np
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def create_word_text_object_topic(word, topic, location, color):
    """Create a Blender text object for the given word, assign a color based on its topic, and add it to the Word Clouds collection."""
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.object
    text_obj.data.body = word

    # Randomize rotation for a "cloudy" appearance
    text_obj.rotation_euler[2] = random.uniform(-0.2, 0.2)

    # Create a new material using the helper function and assign it to the text object
    mat = create_material_from_color(color)
    if text_obj.data.materials:
        text_obj.data.materials[0] = mat
    else:
        text_obj.data.materials.append(mat)

    # Move the object to the "Word Clouds" collection
    word_clouds_coll = get_word_clouds_collection()
    if text_obj.name not in word_clouds_coll.objects:
        word_clouds_coll.objects.link(text_obj)
    # Unlink from any other collections
    for col in text_obj.users_collection:
        if col != word_clouds_coll:
            col.objects.unlink(text_obj)

    return text_obj

def delete_word_cloud():
    if "Word Clouds" in bpy.data.collections:
        coll = bpy.data.collections["Word Clouds"]
        # Delete all objects in the collection
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        # Optionally, remove the collection itself
        bpy.data.collections.remove(coll)

def load_text_file(filepath):
    """Read and return the contents of the text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
def process_text_freq(text):
    """Perform a simple word frequency analysis on the text."""
    # Remove punctuation and make lowercase
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator).lower()
    
    # Split into words and filter out common stopwords
    words = text.split()
    stopwords = {"the", "and", "a", "to", "of", "in", "for", "is", "on", "that", "with", "as", "by", "at"}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count frequency
    counter = Counter(words)
    return counter

def get_word_clouds_collection():
    """Get (or create) a collection named 'Word Clouds'."""
    if "Word Clouds" in bpy.data.collections:
        coll = bpy.data.collections["Word Clouds"]
    else:
        coll = bpy.data.collections.new("Word Clouds")
        bpy.context.scene.collection.children.link(coll)
    return coll

def create_word_text_object(word, frequency, min_frequency, max_frequency, location, low_color, high_color):
    """Create a Blender text object for the given word with size based on frequency,
    assign a color interpolated between low_color and high_color, and add it to the Word Clouds collection."""
    
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.object
    text_obj.data.body = word

    # Set font size proportional to frequency (adjust scaling factor as needed)
    scale_factor = 0.5 + (frequency / max_frequency) * 2.0
    text_obj.data.size = scale_factor

    # Randomize rotation slightly for a "cloudy" appearance
    text_obj.rotation_euler[2] = random.uniform(-0.2, 0.2)

    # Calculate interpolation factor (t = 0 for min frequency, t = 1 for max frequency)
    if max_frequency != min_frequency:
        t = (frequency - min_frequency) / (max_frequency - min_frequency)
    else:
        t = 1.0

    t = t ** (1/3) 
    # Interpolate color between low_color and high_color
    r = low_color[0] * (1 - t) + high_color[0] * t
    g = low_color[1] * (1 - t) + high_color[1] * t
    b = low_color[2] * (1 - t) + high_color[2] * t
    color = (r, g, b, 1.0)

    # Create a new material using the helper function and assign it to the text object
    mat = create_material_from_color(color)
    if text_obj.data.materials:
        text_obj.data.materials[0] = mat
    else:
        text_obj.data.materials.append(mat)

    # Move the object to the "Word Clouds" collection
    word_clouds_coll = get_word_clouds_collection()
    if text_obj.name not in word_clouds_coll.objects:
        word_clouds_coll.objects.link(text_obj)
    # Unlink from any other collections
    for col in text_obj.users_collection:
        if col != word_clouds_coll:
            col.objects.unlink(text_obj)

    return text_obj

def generate_word_cloud(filepath, low_color, high_color, word_limit):
    """Load text file, analyze topics, and create a word cloud in the scene."""
    delete_word_cloud()
    text = load_text_file(filepath)
    if text is None:
        return

    word_counts = process_text_freq(text)
    if not word_counts:
        print("No words found for word cloud.")
        return

    # Limit to the top 50 words for clarity
    most_common_words = word_counts.most_common(word_limit)
    max_frequency = most_common_words[0][1]
    min_frequency = most_common_words[-1][1]

    spread = 10  # Define area for random placement

    # Iterate over the most common words
    for word, frequency in most_common_words:
        location = (random.uniform(-spread, spread),
                    random.uniform(-spread, spread),
                    0)
        create_word_text_object(word, frequency, min_frequency, max_frequency, location, low_color, high_color)

##AXIS-----------------------------------------------------------------------------------------------------------------
def delete_axis(axis):
    """Deletes all objects related to a specific axis."""
    for obj in bpy.data.objects:
        if obj.name.startswith(f"{axis}_Axis") or obj.name.startswith(f"{axis}_Tick"):
            bpy.data.objects.remove(obj, do_unlink=True)

# Helper Function: Create a single axis
def create_axis(context, axis, color):
    """Creates a long rectangle axis with larger ticks and numbering."""
    print("axis creation")
    material = create_material_from_color(color)
    axis_group = bpy.data.collections.get("Axis")
    if not axis_group:
        axis_group = bpy.data.collections.new("Axis")
        bpy.context.scene.collection.children.link(axis_group)
    
    axis_name = f"{axis}_Group"
    axis_collection = bpy.data.collections.get(axis_name)
    if not axis_collection:
        axis_collection = bpy.data.collections.new(axis_name)
        axis_group.children.link(axis_collection)
    
    mesh = bpy.data.meshes.new(f"{axis}_Axis")
    obj = bpy.data.objects.new(f"{axis}_Axis", mesh)
    if obj.type == 'MESH':
        obj.data.materials.append(material)
    axis_collection.objects.link(obj)
    bm = bmesh.new()

    # Define axis size and tick properties
    axis_length = 5
    csv_filepath = getattr(context.scene, "csv_filepath", "")
    if csv_filepath and os.path.exists(csv_filepath):
        max_val = get_max_value_from_csv(csv_filepath)
        if max_val > 0:
            axis_length = int(max_val)
    axis_width = 0.1
    tick_width = 0.2
    tick_height = 0.5
    tick_spacing = 1

    # Create the main axis rectangle
    if axis == "X":
        bm.verts.new((-axis_length, -axis_width, 0))
        bm.verts.new((axis_length, -axis_width, 0))
        bm.verts.new((axis_length, axis_width, 0))
        bm.verts.new((-axis_length, axis_width, 0))
    elif axis == "Y":
        bm.verts.new((-axis_width, -axis_length, 0))
        bm.verts.new((axis_width, -axis_length, 0))
        bm.verts.new((axis_width, axis_length, 0))
        bm.verts.new((-axis_width, axis_length, 0))
    elif axis == "Z":
        bm.verts.new((-axis_width, 0, -axis_length))
        bm.verts.new((axis_width, 0, -axis_length))
        bm.verts.new((axis_width, 0, axis_length))
        bm.verts.new((-axis_width, 0, axis_length))
    
    bm.faces.new(bm.verts)

    # Add ticks and numbering
    for i in range(-axis_length, axis_length + 1, tick_spacing):
        if axis == "X":
            # Create the tick as a rectangle
            verts = [
                bm.verts.new((i - tick_width / 2, axis_width, 0)),
                bm.verts.new((i + tick_width / 2, axis_width, 0)),
                bm.verts.new((i + tick_width / 2, axis_width + tick_height, 0)),
                bm.verts.new((i - tick_width / 2, axis_width + tick_height, 0)),
            ]
            bm.faces.new(verts)

            # Add text for numbering
            font_curve = bpy.data.curves.new(name=f"{axis}_Tick_{i}_Text", type='FONT')
            text_obj = bpy.data.objects.new(name=f"{axis}_Tick_{i}_TextObject", object_data=font_curve)
            text_obj.location = (i, axis_width + tick_height + 0.2, 0)
            text_obj.data.body = str(i)
            text_obj.data.materials.append(material)
            axis_collection.objects.link(text_obj)

        elif axis == "Y":
            verts = [
                bm.verts.new((axis_width, i - tick_width / 2, 0)),
                bm.verts.new((axis_width, i + tick_width / 2, 0)),
                bm.verts.new((axis_width + tick_height, i + tick_width / 2, 0)),
                bm.verts.new((axis_width + tick_height, i - tick_width / 2, 0)),
            ]
            bm.faces.new(verts)

            font_curve = bpy.data.curves.new(name=f"{axis}_Tick_{i}_Text", type='FONT')
            text_obj = bpy.data.objects.new(name=f"{axis}_Tick_{i}_TextObject", object_data=font_curve)
            text_obj.location = (axis_width + tick_height + 0.2, i, 0)
            text_obj.data.body = str(i)
            text_obj.data.materials.append(material)
            axis_collection.objects.link(text_obj)

        elif axis == "Z":
            verts = [
                bm.verts.new((axis_width, 0, i - tick_width / 2)),
                bm.verts.new((axis_width, 0, i + tick_width / 2)),
                bm.verts.new((axis_width + tick_height, 0, i + tick_width / 2)),
                bm.verts.new((axis_width + tick_height, 0, i - tick_width / 2)),
            ]
            bm.faces.new(verts)

            font_curve = bpy.data.curves.new(name=f"{axis}_Tick_{i}_Text", type='FONT')
            text_obj = bpy.data.objects.new(name=f"{axis}_Tick_{i}_TextObject", object_data=font_curve)
            text_obj.location = (axis_width + tick_height + 0.2, 0, i)
            text_obj.data.body = str(i)
            text_obj.data.materials.append(material)
            axis_collection.objects.link(text_obj)

    bm.to_mesh(mesh)
    bm.free()
    return obj

def get_max_value_from_csv(filepath):
    """Reads the CSV file and returns the maximum absolute value found among all coordinates."""
    max_val = 0
    try:
        with open(filepath, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    # Expect each row to have three numbers: x, y, z
                    x = float(row.get('X', 0))  # Default to 0 if 'X' column is missing
                    y = float(row.get('Y', 0))  # Default to 0 if 'Y' column is missing
                    z = float(row.get('Z', 0))  # Default to 0 if 'Z' column is missing
                    max_in_row = max(abs(x), abs(y), abs(z))
                    if max_in_row > max_val:
                        max_val = max_in_row
                except Exception as e:
                    print(f"Skipping row {row} due to error: {e}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    print(max_val)
    return max_val

def create_material_from_color(color):
    mat = bpy.data.materials.new(name="Material")
    mat.diffuse_color = (color)
    return mat

##SCATTERPLOT
def delete_scatter_plot_objects():
    if 'ScatterPlot' in bpy.data.collections:
        collection = bpy.data.collections['ScatterPlot']
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

##LINEPLOT------------------
def delete_line_plot_objects():
    line_collection = bpy.data.collections.get("LineChart")
    if line_collection:
        for obj in line_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
            print("Line plot deleted.")
    else:
        print("No LinePlot collection found.")
    return {'FINISHED'}

def create_line_chart(context):
    delete_line_plot_objects()
    csv_filepath = getattr(context.scene, "csv_filepath", "")
    if not csv_filepath or not os.path.exists(csv_filepath):
        print("CSV file not set or does not exist")
        return {'CANCELLED'}

    points = []
    try:
        with open(csv_filepath, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    # Expect each row to have three numeric values: X, Y, Z
                    x = float(row.get('X', 0))
                    y = float(row.get('Y', 0))
                    z = float(row.get('Z', 0))
                    points.append((x, y, z))
                except Exception as e:
                    print(f"Skipping row {row}: {e}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {'CANCELLED'}

    if not points:
        print("No valid points found in CSV")
        return {'CANCELLED'}

    # Create or get the "LineChart" collection
    line_collection = bpy.data.collections.get("LineChart")
    if line_collection is None:
        line_collection = bpy.data.collections.new("LineChart")
        context.scene.collection.children.link(line_collection)

    # Get the user-specified line color and size from scene properties
    line_color = context.scene.line_plot_color
    line_size = context.scene.line_plot_size

    # Create a material for the line plot
    material = bpy.data.materials.new(name="LinePlot_Material")
    material.diffuse_color = line_color

    # Create line segments between the points using curves for thickness
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        # Create a new Curve data-block
        curve_data = bpy.data.curves.new(name=f"Line_{i}_Curve", type='CURVE')
        curve_data.dimensions = '3D'
        # Set the bevel depth to control thickness
        curve_data.bevel_depth = line_size
        curve_data.fill_mode = 'FULL'
        
        # Create a new spline in that curve and add two points.
        spline = curve_data.splines.new('POLY')
        spline.points.add(1)  # this adds one point (total points becomes 2)
        spline.points[0].co = (p1[0], p1[1], p1[2], 1)
        spline.points[1].co = (p2[0], p2[1], p2[2], 1)

        # Create an object with the curve data
        obj = bpy.data.objects.new(f"Line_{i}", curve_data)
        obj.data.materials.append(material)
        line_collection.objects.link(obj)

    print(f"Line chart created with {len(points) - 1} lines")
    return {'FINISHED'}

#BARCHART----------------------------------------------------------
def delete_bar_chart_objects():
    bar_collection = bpy.data.collections.get("BarChart")
    if bar_collection:
        for obj in bar_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        print("Bar chart deleted.")
    else:
        print("No BarChart collection found.")
    return {'FINISHED'}

def create_bar_chart(context):
    delete_bar_chart_objects()
    csv_filepath = getattr(context.scene, "csv_filepath", "")
    if not csv_filepath or not os.path.exists(csv_filepath):
        print("CSV file not set or does not exist")
        return {'CANCELLED'}

    points = []
    try:
        with open(csv_filepath, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    # Expect each row to have two numeric values: X, Y
                    x = float(row.get('X', 0))
                    y = float(row.get('Y', 0))
                    points.append((x, y))
                except Exception as e:
                    print(f"Skipping row {row}: {e}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {'CANCELLED'}

    if not points:
        print("No valid points found in CSV")
        return {'CANCELLED'}

    # Build a dictionary grouping indices by their x value.
    duplicate_x_indices = {}
    for idx, (x, y) in enumerate(points):
        duplicate_x_indices.setdefault(x, []).append(idx)

    # Identify which point(s) to shift: for each duplicate group,
    # choose the one with the lowest y value.
    indices_to_shift = set()
    for x, indices in duplicate_x_indices.items():
        if len(indices) > 1:
            # Find index with the lowest y value
            lowest_index = min(indices, key=lambda i: points[i][1])
            indices_to_shift.add(lowest_index)

    # Create or get the "BarChart" collection
    bar_collection = bpy.data.collections.get("BarChart")
    if bar_collection is None:
        bar_collection = bpy.data.collections.new("BarChart")
        context.scene.collection.children.link(bar_collection)

    base_color = context.scene.bar_plot_color
    bar_width = context.scene.bar_plot_width

   
    # Create bars (3D cubes) for each point
    for i, (x, y) in enumerate(points):
        # Calculate the bar location.
        # The bar is normally located at (x, y/2, 0),
        # but if this point has a duplicate x and is the one with the lowest y,
        # shift it by +1 along the z axis.
        z_shift = 1 if i in indices_to_shift else 0
        location = (x, y / 2, z_shift)
        variant_color = vary_color(base_color, variation=0.4)
        material = create_material_from_color_with_alpha(variant_color)
        # Create a cube mesh for the bar
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        bar_obj = bpy.context.object
        bar_obj.scale = (bar_width, y, 1)  # Set scale to create a rectangle shape
        bar_obj.name = f"Bar_{i+1}"
        bar_obj.data.materials.append(material)
        
        new_obj = context.active_object
        for coll in new_obj.users_collection:
            coll.objects.unlink(new_obj)
        bar_collection.objects.link(new_obj)

    print(f"Bar chart created with {len(points)} bars")
    return {'FINISHED'}


def vary_color(color, variation=0.2):
    # Check if the color has 4 values (RGBA) or 3 values (RGB)
    if len(color) == 3:
        r, g, b = color
        a = 1  # Default to fully opaque if alpha is not provided
    else:
        raise ValueError("Color must have 3 (RGB) or 4 (RGBA) components.")
    
    r = max(0, min(1, r + random.uniform(-variation, variation)))
    g = max(0, min(1, g + random.uniform(-variation, variation)))
    b = max(0, min(1, b + random.uniform(-variation, variation)))
    
    return (r, g, b, a)
    
def create_material_from_color_with_alpha(color):
    mat = bpy.data.materials.new(name="Material")
    # Ensure the color is a 4-item tuple (RGBA), add an alpha value if missing (default is 1)
    if len(color) == 3:
        color = (*color, 1.0)  # Add alpha value if missing
    mat.diffuse_color = color
    return mat

def compute_histogram(values, num_bins):
    """Computes histogram bin edges and frequencies from a list of numeric values."""
    if not values:
        return None, None
    vmin = min(values)
    vmax = max(values)
   
    if vmax == vmin:
        vmax += 1
    bin_width = (vmax - vmin) / num_bins
   
    bin_edges = [vmin + i * bin_width for i in range(num_bins + 1)]
    frequencies = [0] * num_bins
    for v in values:
        
        bin_index = int((v - vmin) / bin_width)
        if bin_index == num_bins:
            bin_index = num_bins - 1
        frequencies[bin_index] += 1
    return bin_edges, frequencies


def delete_histogram_objects():
    """Deletes all objects in the 'Histogram' collection."""
    hist_collection = bpy.data.collections.get("Histogram")
    if hist_collection:
        for obj in list(hist_collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        print("Histogram objects deleted.")
    else:
        print("No Histogram collection found.")
    return {'FINISHED'}


def create_histogram(context):
    """Reads a CSV file, computes a histogram from the 'Value' column, and creates cubes for each bin."""
    delete_histogram_objects()
    csv_filepath = getattr(context.scene, "csv_filepath", "")
    if not csv_filepath or not os.path.exists(csv_filepath):
        print("CSV file not set or does not exist")
        return {'CANCELLED'}

    # Read CSV values from the column "Value"
    values = []
    try:
        with open(csv_filepath, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    v = float(row.get('X', 0))
                    values.append(v)
                except Exception as e:
                    print(f"Skipping row {row}: {e}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {'CANCELLED'}

    if not values:
        print("No valid numeric values found in CSV for histogram")
        return {'CANCELLED'}

    num_bins = context.scene.hist_bins
    bin_edges, frequencies = compute_histogram(values, num_bins)
    if bin_edges is None:
        print("Histogram computation failed")
        return {'CANCELLED'}

    # Create or get the "Histogram" collection
    hist_collection = bpy.data.collections.get("Histogram")
    if hist_collection is None:
        hist_collection = bpy.data.collections.new("Histogram")
        context.scene.collection.children.link(hist_collection)

    base_color = context.scene.hist_color

    # For each bin, create a cube scaled by frequency.
    for i in range(num_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        frequency = frequencies[i]
        # Place the cube centered in the bin on the X axis.
        x_center = (bin_start + bin_end) / 2
        # Height of the cube corresponds to frequency; its base is at Y = 0.
        height = frequency
        y_center = height / 2

        variant_color = vary_color(base_color, variation=0.3)
        material = create_material_from_color(variant_color)

        bpy.ops.mesh.primitive_cube_add(location=(x_center, y_center, -1))
        cube_obj = bpy.context.object
        # Scale: half the bin width in X, half the frequency in Y (since default cube height is 2), fixed thickness in Z.
        cube_obj.scale = ((bin_end - bin_start) / 2, height / 2, 0.5)
        cube_obj.name = f"Hist_Bin_{i+1}"
        cube_obj.data.materials.append(material)

        # Move the object into the Histogram collection.
        for coll in cube_obj.users_collection:
            coll.objects.unlink(cube_obj)
        hist_collection.objects.link(cube_obj)

    print("Histogram created with {} bins.".format(num_bins))
    return {'FINISHED'}


def vary_color(color, variation=0.2):
    if len(color) == 3:
        r, g, b = color
        a = 1.0
    elif len(color) == 4:
        r, g, b, a = color
    else:
        raise ValueError("Color must be a 3- or 4-item tuple.")
    
    r = max(0, min(1, r + random.uniform(-variation, variation)))
    g = max(0, min(1, g + random.uniform(-variation, variation)))
    b = max(0, min(1, b + random.uniform(-variation, variation)))
    
    return (r, g, b, a)

def process_text_tfidf(text):
    """Perform TF-IDF analysis on the text and return word importance scores."""
    
    stopwords = ["the", "is", "and", "to", "as", "at", "by", "on", "in", "a", "with", "for", "that", "of"]

    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True)

    # Fit and transform the text into TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform([text])

    # Get words and their corresponding TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()

    # Convert to dictionary format {word: score}
    word_tfidf = dict(zip(feature_names, scores))

    return Counter(word_tfidf)

def generate_word_cloud_tfidf(filepath, low_color, high_color, word_limit):
    """Load text file, analyze topics, and create a word cloud in the scene."""
    delete_word_cloud()
    text = load_text_file(filepath)
    if text is None:
        return

    word_counts = process_text_tfidf(text)
    if not word_counts:
        print("No words found for word cloud.")
        return

    # Limit to the top 50 words for clarity
    most_common_words = word_counts.most_common(word_limit)
    max_frequency = most_common_words[0][1]
    min_frequency = most_common_words[-1][1]

    spread = 10  # Define area for random placement

    # Iterate over the most common words
    for word, frequency in most_common_words:
        location = (random.uniform(-spread, spread),
                    random.uniform(-spread, spread),
                    0)
        create_word_text_object(word, frequency, min_frequency, max_frequency, location, low_color, high_color)


def process_text_lda(text, custom_stopwords=None):
    """Preprocess the text by removing stopwords and punctuation."""
    if custom_stopwords is None:
        custom_stopwords = []

    words = text.split()
    processed_words = [word.lower() for word in words if word.lower() not in custom_stopwords and word not in string.punctuation]
    return ' '.join(processed_words)

def generate_topic_model(filepath, n_topics=5, word_limit=50):
    """Load text file, apply LDA, and create word cloud based on topics."""
    delete_word_cloud()
    text = load_text_file(filepath)
    if text is None:
        return

    custom_stopwords = ["and", "the", "to", "a", "of", "in", "for", "on", "with", "that", "this", "an"]

    # Pass the custom stopwords when calling process_text
    processed_text = process_text_lda(text, custom_stopwords)

    # Vectorize the text into a document-term matrix (count vectorization)
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([processed_text])

    # Apply LDA to the document-term matrix
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    # Get the feature names (words) and topic distributions
    words = vectorizer.get_feature_names_out()
    topic_words = {}
    for topic_idx, topic in enumerate(lda.components_):
        topic_words[topic_idx] = [words[i] for i in topic.argsort()[:-word_limit - 1:-1]]

    # Define parameters for clustering topics:
    global_radius = 15  # distance from the origin for each topic cluster center
    local_spread = 2    # how far words are scattered around the topic center
    z_spread = 2        # spread along the Z-axis

    # Arrange topic clusters in a circle
    for topic_idx, words_in_topic in topic_words.items():
        angle = 2 * np.pi * topic_idx / n_topics
        center_x = global_radius * np.cos(angle)
        center_y = global_radius * np.sin(angle)
        
        # Assign a random color for each topic
        color = (random.random(), random.random(), random.random(), 1.0)
        
        for word in words_in_topic:
            location = (center_x + random.uniform(-local_spread, local_spread),
                        center_y + random.uniform(-local_spread, local_spread),
                        random.uniform(-z_spread, z_spread))  # Add variation in Z-axis
            create_word_text_object_topic(word, topic_idx, location, color)