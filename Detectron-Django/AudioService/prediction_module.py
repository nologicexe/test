import matplotlib.pyplot as plt
import os
from bokeh.plotting import figure
from bokeh.io import export_png
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource
import math
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib
# import logging
matplotlib.use('Agg')

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)

# Prediction plot attributes:
PLOT_WIDTH = 1121
PLOT_HEIGHT = 621

# window size, inside of which mel-plots are created
WINDOW_SIZE = 2
N_FFT = 2048
N_MELS = 128
HOP_LENGTH = 512

MODEL_NAME = 'model_cases_sixclasses.hdf5'
MODEL_PATH = os.path.join(os.getcwd(), 'AudioService', 'nn_models', MODEL_NAME)
SAMPLES_PATH = os.path.join(os.getcwd(), 'AudioService','graphs')


def Predict_On_Plot():
    model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    image_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    pred_gen = image_data_gen.flow_from_directory(
        SAMPLES_PATH,
        class_mode="sparse",
        target_size=(320, 240),
        color_mode='rgb',
        shuffle=False)

    predictions = model.predict(pred_gen)
    top_prediction = predictions.argmax(axis=1)
    return top_prediction.tolist()


def Process_File(filename):
    file_predictions = []
    
    result_plot_name = filename.replace(filename[-4:], ".png")

    audio_file_path = os.path.join(
        os.getcwd(), 'AudioService', 'static', 'AudioService', filename)
    plot_file_path = os.path.join(os.getcwd(), 'AudioService', 'graphs', 'to_predict', result_plot_name)

    y, sr = librosa.load(audio_file_path)
    total_duration = math.floor(librosa.get_duration(y, sr))

    offset = 0
    while offset < total_duration:
        audio_segment, sr = librosa.load(
            audio_file_path, offset=offset, duration=WINDOW_SIZE)
        trimmed_samples, _ = librosa.effects.trim(audio_segment)
        spectrosamples = librosa.feature.melspectrogram(trimmed_samples, sr=sr, n_fft=N_FFT,
                                                        hop_length=HOP_LENGTH,
                                                        n_mels=N_MELS)
        spectrosamples_db = librosa.power_to_db(spectrosamples, ref=np.max)
        librosa.display.specshow(spectrosamples_db, sr=sr, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis='mel')
        fig = plt.figure()
        fig.patch.set_visible(False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('tight')
        ax.axis('off')
        librosa.display.specshow(
            spectrosamples_db, sr=sr, x_axis='time', y_axis='hz')

        with open(plot_file_path, mode="wb") as open_file:
            plt.savefig(open_file)
            plt.close('all')

        segment_prediction = Predict_On_Plot()
        file_predictions.append(segment_prediction)

        os.remove(plot_file_path)

        offset += WINDOW_SIZE

    
    Build_ResultPlot(plot_filename=os.path.join(os.getcwd(), 'AudioService', 'static', 'AudioService', result_plot_name), predictions_list=file_predictions,
                     plot_duration=total_duration)
    


def Build_ResultPlot(plot_filename, predictions_list, plot_duration):
    labels_dict_reduced = {
        0: "Angry",
        1: "Calm",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad"
    }

    labels = list(labels_dict_reduced.values())

    predicted_labels = []

    for y_label in predictions_list:
        predicted_labels.append(labels_dict_reduced.get(y_label[0]))

    glyphs_start_coords = list(range(0, plot_duration, WINDOW_SIZE))
    glyphs_end_coords = [x+WINDOW_SIZE for x in glyphs_start_coords]

    source = ColumnDataSource(data=dict(predicted_labels=predicted_labels,
                                        left_coords=glyphs_start_coords, right_coords=glyphs_end_coords))

    p = figure(y_range=labels, x_range=(0, plot_duration), plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, title="Emotions",
               toolbar_location=None, tools="")

    p.hbar(y='predicted_labels', left='left_coords',
           right='right_coords', width=0.9, source=source, fill_color=factor_cmap('predicted_labels', palette=Spectral6, factors=labels), line_color=None)

    p.xgrid.grid_line_color = None
    p.outline_line_color = None

    export_png(p, filename=str(plot_filename))
