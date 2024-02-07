import streamlit as st
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
#from sklearn.metrics import silhouette_score  # Import silhouette_score
import pandas as pd
from matplotlib.patches import Patch
from GenderIdentifier import GenderIdentifier
from pydub import AudioSegment
from playsound import playsound

# Replace 'your_audio_file.mp3' with the path to your audio file


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>SPEAKER DIARIZATION</h1>", unsafe_allow_html=True)

segment_path = r'd:/major_project/audio/segments'

#uploaded_file = st.file_uploader("Choose a file")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames, placeholder='Select a file')
    return os.path.join(folder_path, selected_filename)

filename = file_selector(folder_path=r'd:/major_project/audio/wav_files/')
name_of_file = os.path.basename(filename)
#print(uploaded_file)
def del_prev_graph(graph):
    files = os.listdir(r'D:\major_project\audio\output')
    for file in files:
        if graph in file:
            os.remove(os.path.join(r'D:\major_project\audio\output', file))

def wave_graph(filename, name_of_file = name_of_file):
    files = os.listdir(r'D:\major_project\audio\output')
    if name_of_file + '_wave.png' in files:
        example_image = plt.imread(rf'D:\major_project\audio\output\{name_of_file}_wave.png')
        st.image(example_image, use_column_width=True)
    else:
        del_prev_graph('wave')
        y, sr = librosa.load(filename)
        time = librosa.times_like(y)
        fig = plt.figure(figsize=(16, 5))
        plt.plot(y, label='Data')
        plt.title('Waveform of the Audio Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(rf'D:\major_project\audio\output\{name_of_file}_wave.png')
        st.pyplot(fig)

def get_y(path):
    y, sr = librosa.load(path)
    return y, sr

def derivative_graph(y, threshold=0.02, name_of_file=name_of_file):
    files = os.listdir(r'D:\major_project\audio\output')
    data = y
    derivative = np.gradient(data)
    sudden_changes = np.where(np.abs(derivative) > threshold)[0]
    if name_of_file + '_derivative_' + str(threshold) + '.png' in files:
        example_image = plt.imread(rf'D:\major_project\audio\output\{name_of_file}_derivative_{threshold}.png')
        st.image(example_image, use_column_width=True)
    else:
        del_prev_graph('derivative')
        data = y
        derivative = np.gradient(data)
        fig = plt.figure(figsize=(16,5))
        plt.plot(derivative, label='Derivative')
        plt.title('Derivative (Change)')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        sudden_changes = np.where(np.abs(derivative) > threshold)[0]
        plt.scatter(sudden_changes, derivative[sudden_changes], color='red', label='Sudden Change')
        plt.legend()
        plt.tight_layout()
        plt.savefig(rf'D:\major_project\audio\output\{name_of_file}_derivative_{threshold}.png')
        st.pyplot(fig)
    return sudden_changes

def show_selection(y, li, threshold):
    files = os.listdir(r'D:\major_project\audio\output')
    if name_of_file + '_selection_' + str(threshold) + '.png' in files:
        example_image = plt.imread(rf'D:\major_project\audio\output\{name_of_file}_selection_{threshold}.png')
        st.image(example_image,use_column_width=True)
    else:
        del_prev_graph('selection')
        data = y
        fig = plt.figure(figsize=(16, 5))
        plt.plot(data, label='Data')
        for i, (start, end) in enumerate(li):
            color = plt.cm.viridis(i / len(li))
            plt.axvspan(start, end, alpha=0.3, color=color, label=f'Highlighted Region {i + 1} ({start}-{end})')
            plt.savefig(rf'D:\major_project\audio\output\{name_of_file}_selection_{threshold}.png')
        st.pyplot(fig)

def show_selection_graph(sudden_changes, length_threshold = 6000, lower_offset = 0, upper_offset = 3000, threshold = 0.02):
    lower_limit = 0
    count = 0
    li = []
    for i in range(len(sudden_changes)-1):
        if sudden_changes[i+1] - sudden_changes[i] > 6000:
            upper_limit = sudden_changes[i]
            y_save = y[lower_limit : upper_limit+ upper_offset]
            if len(y_save)/sr > 0.5:
                t1 = (lower_limit, upper_limit)
                li.append(t1)
                count = count+1
            lower_limit = sudden_changes[i+1]
    y_save = y[upper_limit:]
    t1 = (upper_limit , len(y))
    li.append(t1)
    show_selection(y, li, threshold)
    return li

def segment(sudden_changes, output_path, length_threshold = 6000, lower_offset = 0, upper_offset = 3000):
    lower_limit = 0
    count = 0
    li = []
    for i in range(len(sudden_changes)-1):
        if sudden_changes[i+1] - sudden_changes[i] > 6000:
            upper_limit = sudden_changes[i]
            y_save = y[lower_limit : upper_limit+ upper_offset]
            if len(y_save)/sr > 0.5:
                t1 = (lower_limit, upper_limit)
                li.append(t1)
                sf.write(fr'{output_path}/segment{count}.wav', y_save, sr)
                count = count+1
            lower_limit = sudden_changes[i+1]
    y_save = y[upper_limit:]
    t1 = (upper_limit , len(y))
    li.append(t1)
    sf.write(fr'{output_path}/segment{count}.wav', y_save, sr)

def delete_files(path):
    folder_path = path
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


y, sr = get_y(filename)
wave_graph(filename, name_of_file)
number = st.number_input('Set change threshold (Defaut: 0.02):', value=0.02, step = 0.005, format='%.3f')
sudden_changes = derivative_graph(y, threshold=number)
li = show_selection_graph(sudden_changes, threshold=number)


def cluster_and_label(segment_path, fs = 1):
    len_list = []
    files = os.listdir(segment_path)
    for i in range(len(files)):
        audio_path = fr'd:/major_project/audio/segments/segment{i}.wav'
        y, sr = librosa.load(audio_path)
        len_list.append(len(y))
    min_len = min(len_list)
    mfcc_list = []
    pitch_list = []
    files = os.listdir(segment_path)
    for i in range(len(files)):
        audio_path = fr'd:/major_project/audio/segments/segment{i}.wav'
        y, sr = librosa.load(audio_path)
        original_length = len(y)
        specified_length = min_len
        trim_size = (original_length - specified_length) // 2
    
    # Trim the original list
        trimmed_y = y[trim_size:original_length - trim_size]
        #y = trimmed_y
        pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        avg_pitch = np.mean(pitch)
        avg_list = []
        for mfcc in mfccs:
            avg_list.append(np.mean(mfcc))
        mean_mfcc = np.mean(avg_list)
        pitch_list.append(avg_pitch)
        mfcc_list.append(mean_mfcc)
    df = pd.DataFrame({'pitch': pitch_list,'mfcc': mfcc_list})
    if fs == 1:
        feature = 'mfcc'
    else:
        feature = 'pitch'
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    df['pitch'] = ss.fit_transform(df['pitch'].values.reshape(-1,1))
    data = np.array(df[feature])
    data = data.reshape(-1, 1)
    linkage_matrix = linkage(data, method='ward')

    threshold = 1.2 
    #fig = plt.figure(figsize=(16, 5))
    #dendrogram(linkage_matrix)
    #plt.axhline(y=threshold, color='r', linestyle='--')
    #plt.title('Hierarchical Clustering Dendrogram')
    #st.pyplot(fig)

    n_clusters = len(np.unique(cut_tree(linkage_matrix, height=threshold)))
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg_cluster.fit_predict(data)
    clusters = labels.reshape(-1)
    return clusters

def show_cluster(y, li, arr):
    data = y
    fig = plt.figure(figsize=(16, 5))
    plt.plot(data, label='Data')
    unique_numbers = np.unique(arr)
    unique_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_numbers)))
    number_to_color = {num: color for num, color in zip(unique_numbers, unique_colors)}
    handles = []
    legend_labels = set()
    for i, (start, end) in enumerate(li):
        color = number_to_color[arr[i]]
        label = f'Speaker {arr[i]+1}'
        plt.axvspan(start, end, alpha=0.3, color=color, label=f'Highlighted Region {label} ({start}-{end})')
        if label not in legend_labels:
            handles.append(Patch(color=color, label=label))
            legend_labels.add(label)
    plt.legend(handles=handles)        
    st.pyplot(fig)


def show_named_cluster(y, li, arr, names):
    name_list = speaker_naming(arr, names)
    data = y
    fig = plt.figure(figsize=(16, 5))
    plt.plot(data, label='Data')
    unique_numbers = np.unique(name_list)
    unique_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_numbers)))
    number_to_color = {num: color for num, color in zip(unique_numbers, unique_colors)}
    handles = []
    legend_labels = set()
    for i, (start, end) in enumerate(li):
        color = number_to_color[name_list[i]]
        label = name_list[i]
        plt.axvspan(start, end, alpha=0.3, color=color, label=f'Highlighted Region {label} ({start}-{end})')
        if label not in legend_labels:
            handles.append(Patch(color=color, label=label))
            legend_labels.add(label)
    plt.legend(handles=handles)         
    st.pyplot(fig)

if st.button("Segment"):
    delete_files(segment_path)
    segment(sudden_changes, segment_path)
    st.session_state.script_run = True
    files = os.listdir(segment_path)

    st.success(f'Segmentation Completed. {len(files)} segments created.', icon="✅")

fs = st.selectbox('Select Feature', ['mfcc(recommended)', 'pitch'])
if fs == 'mfcc(recommended)':
    fs = 1
else:
    fs = 0

def speaker_naming(arr, names):
    speaker = arr
    name = names
    li = []
    for i in speaker:
        if i not in li:
            li.append(i)
    mapping = {}
    for i in range(len(li)):
        mapping[li[i]] = name[i]
    result_list = [mapping[item] for item in speaker]
    return result_list

def join_segments(arr, output_path, segment_path, names):
    #print(arr)
    #name_list = speaker_naming(arr, names)
    files = os.listdir(segment_path)
    if len(files) == 1:
        y,sr = librosa.load(fr'{segment_path}/segment0.wav')
        sf.write(fr'{output_path}/speaker_1.wav', y, sr)
        return
    y_list = np.empty(1)
    speaker = 1
    #speaker_name = np.unique(name_list)
    for i in np.unique(arr):
        y_list = np.empty(1)
        for j in range(len(arr)):
            if arr[j] == i:
                y, sr = librosa.load(fr'{segment_path}/segment{j}.wav')
                y_list = np.append(y_list, y)
        sf.write(fr'{output_path}/speaker_{speaker}.wav', y_list, sr)
        speaker += 1  

def split_audio(input_file, output_folder, num_segments=1):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the length of each segment
    segment_length = len(audio) // num_segments

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]

    delete_files(output_folder)

    # Split the audio into segments
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length

        # Extract the segment
        segment = audio[start_time:end_time]

        # Save the segment to a new file with the original file name
        output_file = os.path.join(output_folder, f'{file_name}_{i + 1}.wav')
        segment.export(output_file, format="wav")

    print(f"{num_segments} segments created successfully.")

if st.button("Cluster and Label"):
    speakers = cluster_and_label(segment_path, fs)
    st.success(f'{len(np.unique(speakers))} speakers detected', icon="✅")
    show_cluster(y, li, speakers)

if st.button('play audio'):
    files = os.listdir(segment_path)
    speakers = cluster_and_label(segment_path, fs)
    print(speakers)
    i = 0
    for file in files:
        st.write(f'Current speaker: Speaker_{speakers[i]+1}')
        playsound(fr'{segment_path}/{file}')
        i+=1

#user_input_string = st.text_input("Enter names of speakers in Order:")
user_input_string = 'a,b,c'

if st.button('Predict gender'):
    speakers = cluster_and_label(segment_path, fs)
    user_inputs = user_input_string.split(",")
    names = user_inputs
    #show_named_cluster(y, li, speakers, names)
    
    joined_segments_path = r'd:\major_project\audio\joined_segments'
    delete_files(joined_segments_path)
    join_segments(speakers, joined_segments_path, segment_path, names)
    output_segment_path = r'D:\major_project\gender identification\temp'
    files = os.listdir(joined_segments_path)
    for file in files:
        split_audio(fr'{joined_segments_path}/{file}', output_segment_path)
        final_segments = os.listdir(output_segment_path)
        gender_identifier = GenderIdentifier(output_segment_path, 'females.gmm', 'males.gmm')
        gender_identifier.process()
    #GenderIdentifier(joined_segments_path, 'females.gmm', 'males.gmm').process()
    #print(speaker_naming(speakers, names), speakers)


