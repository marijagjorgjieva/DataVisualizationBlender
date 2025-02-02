import bpy
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

def register_axis_color_properties():
    bpy.types.Scene.x_axis_color = bpy.props.FloatVectorProperty(
        name="X Axis Color", subtype='COLOR', size=4, default=(1, 0, 0, 1)  # Red by default
    )
    bpy.types.Scene.y_axis_color = bpy.props.FloatVectorProperty(
        name="Y Axis Color", subtype='COLOR', size=4, default=(0, 1, 0, 1)  # Green by default
    )
    bpy.types.Scene.z_axis_color = bpy.props.FloatVectorProperty(
        name="Z Axis Color", subtype='COLOR', size=4, default=(0, 0, 1, 1)  # Blue by default
    )
    bpy.types.Scene.line_plot_color = bpy.props.FloatVectorProperty(
        name="Line Plot Color", subtype='COLOR', size=4, default=(1, 1, 1, 1)  # White by default
    )
    bpy.types.Scene.line_plot_size = bpy.props.FloatProperty(
        name="Line Size", default=0.1, min=0, description="Thickness of the line plot"
    )