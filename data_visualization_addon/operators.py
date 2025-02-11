import bpy
from .helpers import *
import os
import csv
import bmesh
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



class OBJECT_OT_AddTopicCloud(bpy.types.Operator):
    """Generate and add topic clouds using the imported text file"""
    bl_idname = "object.add_topic_cloud"
    bl_label = "Add Topic Clouds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        filepath = context.scene.get("text_filepath", "")
        if not filepath:
            self.report({'ERROR'}, "No text file imported. Please import a text file first.")
            return {'CANCELLED'}
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        word_limit = context.scene.word_limit
        n_topics = context.scene.n_topics

        generate_topic_model(filepath, n_topics, word_limit)
        self.report({'INFO'}, "Topic clouds generated")
        return {'FINISHED'}

class OBJECT_OT_DeleteTopicCloud(bpy.types.Operator):
    """Delete all objects in the Word Clouds collection"""
    bl_idname = "object.delete_topic_cloud"
    bl_label = "Delete Topic Clouds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_word_cloud()            
        return {'FINISHED'}

class OBJECT_OT_AddWordCloudTFIDF(bpy.types.Operator):
    """Generate and add word clouds using the imported text file"""
    bl_idname = "object.add_wordcloudtfidf"
    bl_label = "Add Word Clouds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        filepath = context.scene.get("text_filepath", "")
        if not filepath:
            self.report({'ERROR'}, "No text file imported. Please import a text file first.")
            return {'CANCELLED'}
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        # Retrieve the user-selected colors from the scene
        low_color = context.scene.low_frequency_color
        high_color = context.scene.high_frequency_color

        word_limit = context.scene.word_limit

        generate_word_cloud_tfidf(filepath, low_color, high_color, word_limit)
        self.report({'INFO'}, "Word clouds generated")
        return {'FINISHED'}

class OBJECT_OT_DeleteWordCloudTFIDF(bpy.types.Operator):
    """Delete all objects in the Word Clouds collection"""
    bl_idname = "object.delete_wordcloudtfidf"
    bl_label = "Delete Word Clouds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_word_cloud()            
        return {'FINISHED'}
    
class OBJECT_OT_CreateHistogram(bpy.types.Operator):
    bl_idname = "object.create_histogram"
    bl_label = "Create Histogram"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return create_histogram(context)


class OBJECT_OT_DeleteHistogram(bpy.types.Operator):
    bl_idname = "object.delete_histogram"
    bl_label = "Delete Histogram"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return delete_histogram_objects()
    
class OBJECT_OT_DeleteBarChart(bpy.types.Operator):
    bl_idname = "object.delete_bar_chart"
    bl_label = "Delete Bar Chart"

    def execute(self, context):
        delete_bar_chart_objects()
        return {'FINISHED'}

class OBJECT_OT_CreateBarChart(bpy.types.Operator):
    bl_idname = "object.create_bar_chart"
    bl_label = "Create Bar Chart"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return create_bar_chart(context)

class OBJECT_OT_DeleteLinePlot(bpy.types.Operator):
    bl_idname = "object.delete_line_plot"
    bl_label = "Delete Line Plot"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_line_plot_objects()
        return {'FINISHED'}

class OBJECT_OT_CreateLinePlot(bpy.types.Operator):
    bl_idname = "object.create_line_plot"
    bl_label = "Create Line Plot"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
       return create_line_chart(context)
       return {'FINISHED'}
    

class OBJECT_OT_AddXAxis(bpy.types.Operator):
    bl_idname = "object.add_x_axis"
    bl_label = "Add X Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        x_color = context.scene.x_axis_color
        delete_axis("X")
        create_axis(context, axis="X",color=x_color)
        return {'FINISHED'}

class OBJECT_OT_AddYAxis(bpy.types.Operator):
    bl_idname = "object.add_y_axis"
    bl_label = "Add Y Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        y_color = context.scene.y_axis_color
        delete_axis("Y")
        create_axis(context, axis="Y",color=y_color)
        return {'FINISHED'}

class OBJECT_OT_AddZAxis(bpy.types.Operator):
    bl_idname = "object.add_z_axis"
    bl_label = "Add Z Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        z_color = context.scene.z_axis_color
        delete_axis("Z")
        create_axis(context, axis="Z",color=z_color)
        return {'FINISHED'}

class OBJECT_OT_DeleteXAxis(bpy.types.Operator):
    bl_idname = "object.delete_x_axis"
    bl_label = "Delete X Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_axis("X")
        return {'FINISHED'}

class OBJECT_OT_DeleteYAxis(bpy.types.Operator):
    bl_idname = "object.delete_y_axis"
    bl_label = "Delete Y Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_axis("Y")
        return {'FINISHED'}

class OBJECT_OT_DeleteZAxis(bpy.types.Operator):
    bl_idname = "object.delete_z_axis"
    bl_label = "Delete Z Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_axis("Z")
        return {'FINISHED'}
    
class OBJECT_OT_ImportCsv(bpy.types.Operator):
    bl_idname = "object.import_csv"
    bl_label = "Import CSV"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "Filepath not set")
            return {'CANCELLED'}

        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}

        context.scene.csv_filename = os.path.basename(self.filepath)
        context.scene.csv_filepath = self.filepath
        self.import_csv(context, self.filepath)
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def import_csv(self, context, filepath):
        pass
    
class OBJECT_OT_DeleteScatterPlot(bpy.types.Operator):
    bl_idname = "object.delete_scatter_plot"
    bl_label = "Delete Scatter plot"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_scatter_plot_objects()
        return {'FINISHED'}

class OBJECT_OT_CreateScatterPlot(bpy.types.Operator):
    bl_idname = "object.create_scatter_plot"
    bl_label = "Create Scatter Plot"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_scatter_plot_objects()
        csv_filepath = getattr(context.scene, "csv_filepath", "")
        if not csv_filepath or not os.path.exists(csv_filepath):
            self.report({'ERROR'}, "CSV file not set or does not exist")
            return {'CANCELLED'}

        color = context.scene.scatter_plot_color

        # Create a material for the line plot
        material = bpy.data.materials.new(name="ScatterPlot_Material")
        material.diffuse_color = color
        points = []
        labels = []
        unique_labels = set()
        try:
            with open(csv_filepath, newline='') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                col_x = context.scene.csv_column_x
                col_y = context.scene.csv_column_y
                col_z = context.scene.csv_column_z
                for row in csv_reader:
                    try:
                        if len(row) >= 2:
                            x = float(row.get(col_x, 0))  # Default to 0 if 'X' column is missing
                            y = float(row.get(col_y, 0))  # Default to 0 if 'Y' column is missing
                            z = float(row.get(col_z, 0))  # Default to 0 if 'Z' column is missing
                            label = row.get('Label', 'default')
                            points.append((x, y, z))
                            labels.append(label)
                            
                    except Exception as e:
                        self.report({'WARNING'}, f"Skipping row {row}: {e}")
        except Exception as e:
            self.report({'ERROR'}, f"Error reading CSV: {e}")
            return {'CANCELLED'}

        if not points:
            self.report({'WARNING'}, "No valid points found in CSV")
            return {'CANCELLED'}
        
        scatter_collection = bpy.data.collections.get("ScatterPlot")
        if scatter_collection is None:
            scatter_collection = bpy.data.collections.new("ScatterPlot")
            context.scene.collection.children.link(scatter_collection)
        
        for point in points:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=point)
            new_obj = context.active_object
            new_obj.data.materials.append(material)
            for coll in new_obj.users_collection:
                coll.objects.unlink(new_obj)
            scatter_collection.objects.link(new_obj)

        self.report({'INFO'}, f"Scatter plot created with {len(points)} points")
        return {'FINISHED'}
    

class OBJECT_OT_ImportTextFile(bpy.types.Operator):
    """Import a text file for word cloud generation"""
    bl_idname = "object.import_text_file"
    bl_label = "Import Text File"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "Filepath not set")
            return {'CANCELLED'}
        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}

        # Save the filepath and filename in the scene for later use
        context.scene.text_filepath = self.filepath
        context.scene.text_filename = os.path.basename(self.filepath)
        self.report({'INFO'}, f"Text file imported: {context.scene.text_filename}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class OBJECT_OT_AddWordCloud(bpy.types.Operator):
    """Generate and add word clouds using the imported text file"""
    bl_idname = "object.add_wordcloud"
    bl_label = "Add Word Clouds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        filepath = context.scene.get("text_filepath", "")
        if not filepath:
            self.report({'ERROR'}, "No text file imported. Please import a text file first.")
            return {'CANCELLED'}
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        # Retrieve the user-selected colors from the scene
        low_color = context.scene.low_frequency_color
        high_color = context.scene.high_frequency_color

        word_limit = context.scene.word_limit

        generate_word_cloud(filepath, low_color, high_color, word_limit)
        self.report({'INFO'}, "Word clouds generated")
        return {'FINISHED'}

class OBJECT_OT_DeleteWordCloud(bpy.types.Operator):
    """Delete all objects in the Word Clouds collection"""
    bl_idname = "object.delete_wordcloud"
    bl_label = "Delete Word Clouds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        delete_word_cloud()            
        return {'FINISHED'}
