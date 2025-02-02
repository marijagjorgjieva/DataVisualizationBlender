bl_info = {
    "name": "Data Visualisaton",
    "blender": (4, 3, 0),
    "category": "Object",
    "description": " ",
}
import random
import bpy
import bmesh
import os
import csv
from .properties import *
from .helpers import *
from .operators import *
from .panels import *
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

classes = [
    # Operators
    OBJECT_OT_CreateHistogram,
    OBJECT_OT_DeleteHistogram,
    OBJECT_OT_CreateLinePlot,
    OBJECT_OT_CreateBarChart,
    OBJECT_OT_AddXAxis,
    OBJECT_OT_AddYAxis,
    OBJECT_OT_AddZAxis,
    OBJECT_OT_DeleteXAxis,
    OBJECT_OT_DeleteYAxis,
    OBJECT_OT_DeleteZAxis,
    OBJECT_OT_ImportCsv,
    OBJECT_OT_DeleteLinePlot,
    OBJECT_OT_DeleteBarChart,
    OBJECT_OT_CreateScatterPlot,
    OBJECT_OT_DeleteScatterPlot,
    OBJECT_OT_AddWordCloudTFIDF,
    OBJECT_OT_DeleteWordCloudTFIDF,
    OBJECT_OT_ImportTextFile,
    OBJECT_OT_AddWordCloud,
    OBJECT_OT_DeleteWordCloud,
    OBJECT_OT_AddTopicCloud,
    OBJECT_OT_DeleteTopicCloud,
    
    
    # Panels
    VIEW3D_PT_DataVisualizationPanel,
    VIEW3D_PT_AxisCreatorPanel,
    VIEW3D_PT_CsvImporterPanel,
    VIEW3D_PT_ScatterPlotPanel,
    VIEW3D_PT_LinePlotPanel,
    VIEW3D_PT_BarChartPanel,
    VIEW3D_PT_HistogramPanel,
    VIEW3D_PT_TextFileImporter,
    VIEW3D_PT_WordCloudsFreq,
    VIEW3D_PT_WordCloudsTFIDF,
    VIEW3D_PT_WordCloudsTopics,
]

def register():
    register_axis_color_properties()
    bpy.types.Scene.csv_filename = bpy.props.StringProperty()
    bpy.types.Scene.csv_filepath = bpy.props.StringProperty(subtype="FILE_PATH")
    bpy.types.Scene.text_filename = bpy.props.StringProperty(name="Text Filename", default="")
    bpy.types.Scene.text_filepath = bpy.props.StringProperty(subtype="FILE_PATH", name="Text Filepath", default="")
    bpy.types.Scene.bar_plot_color = bpy.props.FloatVectorProperty(name="Bar Color", subtype='COLOR', default=(0, 1, 0), min=0, max=1)
    bpy.types.Scene.bar_plot_width = bpy.props.FloatProperty(name="Bar Width", default=0.5, min=0.5)
    bpy.types.Scene.hist_color = bpy.props.FloatVectorProperty(
        name="Histogram Color", subtype='COLOR', size=4, default=(0, 0.5, 1, 1)
    )
    bpy.types.Scene.hist_bins = bpy.props.IntProperty(
        name="Histogram Bins", default=10, min=1, max=100
    )
    bpy.types.Scene.low_frequency_color = bpy.props.FloatVectorProperty(
        name="Low Frequency Color",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),  # Default blue
        min=0.0, max=1.0,
        description="Color for words with the lowest frequency"
    )
    bpy.types.Scene.high_frequency_color = bpy.props.FloatVectorProperty(
        name="High Frequency Color",
        subtype='COLOR',
        default=(1.0, 0.0, 0.0),  # Default red
        min=0.0, max=1.0,
        description="Color for words with the highest frequency"
    )
    
    bpy.types.Scene.word_limit = bpy.props.IntProperty(
        name="Word Limit",
        default=50,
        min=1,
        max=200,
        description="Number of words to extract for the word cloud"
    )
    bpy.types.Scene.n_topics = bpy.props.IntProperty(
        name="Number of Topics",
        default=5,
        min=1,
        max=20,
        description="Number of topics to extract"
    )
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    del bpy.types.Scene.csv_filename
    del bpy.types.Scene.csv_filepath
    del bpy.types.Scene.text_filepath
    del bpy.types.Scene.text_filename
    del bpy.types.Scene.x_axis_color
    del bpy.types.Scene.y_axis_color
    del bpy.types.Scene.z_axis_color
    del bpy.types.Scene.line_plot_color
    del bpy.types.Scene.line_plot_size
    del bpy.types.Scene.bar_plot_color
    del bpy.types.Scene.bar_plot_width
    del bpy.types.Scene.hist_color
    del bpy.types.Scene.hist_bins
    del bpy.types.Scene.low_frequency_color
    del bpy.types.Scene.high_frequency_color
    del bpy.types.Scene.word_limit
    del bpy.types.Scene.n_topics

    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()