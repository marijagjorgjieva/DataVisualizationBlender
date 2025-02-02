import bpy
from .operators import *
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


class VIEW3D_PT_HistogramPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_histogram"
    bl_label = "Histogram Creator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Histogram'
    bl_parent_id = "OBJECT_PT_csv_importer"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "hist_color", text="Histogram Color")
        layout.prop(scene, "hist_bins", text="Number of Bins")
        row = layout.row()
        row.operator(OBJECT_OT_CreateHistogram.bl_idname, text="Create Histogram")
        row.operator(OBJECT_OT_DeleteHistogram.bl_idname, text="Delete Histogram")

class VIEW3D_PT_BarChartPanel(bpy.types.Panel):
    bl_label = "Bar Chart Creator"
    bl_idname = "VIEW3D_PT_BarChartPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Bar Chart'
    bl_parent_id = "OBJECT_PT_csv_importer" 

    def draw(self, context):
        layout = self.layout

        layout.prop(context.scene, "bar_plot_color")
        layout.prop(context.scene, "bar_plot_width")

        layout.operator("object.create_bar_chart")
        layout.operator("object.delete_bar_chart")

class VIEW3D_PT_DataVisualizationPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_data_visualization"
    bl_label = "Data Visualization"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'

    def draw(self, context):
        layout = self.layout
        layout.label(text="See below the functionalities:")
        

# Subpanel (Axis Creator)
class VIEW3D_PT_AxisCreatorPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_axis_creator"
    bl_label = "Axis Creator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "VIEW3D_PT_data_visualization"  

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        row = layout.row()
        row.label(text="Add:")
        row.operator(OBJECT_OT_AddXAxis.bl_idname, text="X Axis")
        row.operator(OBJECT_OT_AddYAxis.bl_idname, text="Y Axis")
        row.operator(OBJECT_OT_AddZAxis.bl_idname, text="Z Axis")
        layout.separator()
        row = layout.row()
        row.label(text="Delete:")
        row.operator(OBJECT_OT_DeleteXAxis.bl_idname, text="X Axis")
        row.operator(OBJECT_OT_DeleteYAxis.bl_idname, text="Y Axis")
        row.operator(OBJECT_OT_DeleteZAxis.bl_idname, text="Z Axis")
        
        layout.separator()
        row = layout.row()
        row.label(text="Select Axis Colors:")
        layout.prop(scene, "x_axis_color", text="X Axis Color")
        layout.prop(scene, "y_axis_color", text="Y Axis Color")
        layout.prop(scene, "z_axis_color", text="Z Axis Color")
        
# Subpanel (CSV importer)
class VIEW3D_PT_CsvImporterPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_csv_importer"
    bl_label = "Default data visualization CSV"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "VIEW3D_PT_data_visualization" 

    def draw(self, context):
        layout = self.layout
        layout.label(text="Data visualizations for CSV files in format X,Y,(Z)")
        layout.separator()

        layout.operator("object.import_csv")
        
        csv_filename = context.scene.csv_filename if hasattr(context.scene, 'csv_filename') else "No file selected"
        layout.label(text=f"Selected file: {csv_filename}")
        
class VIEW3D_PT_ScatterPlotPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_scatter_plot"
    bl_label = "Scatter Plot"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "OBJECT_PT_csv_importer"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.create_scatter_plot", text="Apply Scatter Plot")
        layout.operator("object.delete_scatter_plot", text="Delete Scatter Plot")

class VIEW3D_PT_LinePlotPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_line_plot"
    bl_label = "Line Plot"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "OBJECT_PT_csv_importer"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "line_plot_color", text="Line Plot Color")
        layout.prop(scene, "line_plot_size", text="Line Size")
        layout.operator("object.create_line_plot", text="Apply Line Plot")
        layout.operator("object.delete_line_plot", text="Delete Line Plot")


class VIEW3D_PT_WordCloudsFreq(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_word_f_clouds"
    bl_label = "Word Frequency Clouds"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "VIEW3D_PT_text_file_importer"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.add_wordcloud", text="Add Word Frequency Clouds")
        layout.operator("object.delete_wordcloud", text="Delete Word Frequency Clouds")
        
        layout.prop(context.scene, "low_frequency_color", text="Low Frequency Color")
        layout.prop(context.scene, "high_frequency_color", text="High Frequency Color")


class VIEW3D_PT_WordCloudsTFIDF(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_word_tfidf_clouds"
    bl_label = "Word Clouds TFIDF"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "VIEW3D_PT_text_file_importer"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.add_wordcloudtfidf", text="Add Word TFIDF Clouds")
        layout.operator("object.delete_wordcloudtfidf", text="Delete Word TFIDF Clouds")
    
        layout.prop(context.scene, "low_frequency_color", text="Low Frequency Color")
        layout.prop(context.scene, "high_frequency_color", text="High Frequency Color")

class VIEW3D_PT_WordCloudsTopics(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_word_lda_clouds"
    bl_label = "Word Clouds Topics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "VIEW3D_PT_text_file_importer"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.add_topic_cloud", text="Add Topic Clouds")
        layout.operator("object.delete_topic_cloud", text="Delete Topic Clouds")

        layout.prop(context.scene, "n_topics", text="Number of Topics")

class VIEW3D_PT_TextFileImporter(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_text_file_importer"
    bl_label = "NLP data visualization CSV"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Visualization'
    bl_parent_id = "VIEW3D_PT_data_visualization"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Data visualizations for text files")
        layout.separator()

        layout.operator("object.import_text_file", text="Import Text File")
        
        text_filename = getattr(context.scene, "text_filename", "No file selected")
        layout.label(text=f"Selected file: {text_filename}")
        layout.separator()

        layout.prop(context.scene, "word_limit", text="Words per Topic")
