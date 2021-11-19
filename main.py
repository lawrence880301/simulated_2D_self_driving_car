import tkinter as tk
from tkinter import filedialog
from DataPreprocessor import Datapreprocessor
from MLPerceptron import *
from simple_playground import *

model = None
def openfile():
    dataset_url = filedialog.askopenfilename(title="Select file")
    return dataset_url

def draw_track():
    track_data = Datapreprocessor.readfile(openfile())
    track_data = [data.strip().split(',') for data in track_data]
    track_data = Datapreprocessor.text_to_numlist(track_data)
    coordinate_data = track_data[3:]
    x0,y0,x1,y1 = 0, 0, 0, 0
    for idx in range(len(coordinate_data)-1):
        if(idx!= len(coordinate_data) -1):
            x0, y0 = coordinate_data[idx][0] + 100, -coordinate_data[idx][1] + 100
            x1, y1 = coordinate_data[idx+1][0] + 100, -coordinate_data[idx+1][1] + 100
            print(x0, y0, x1, y1)
            canvas.create_line(x0, y0, x1, y1)
        else:
            x0, y0 = coordinate_data[idx][0] + 100, -coordinate_data[idx][1] + 100
            x1, y1 = coordinate_data[0][0] + 100, -coordinate_data[0][1] + 100
            canvas.create_line(x0, y0, x1, y1)

def select_dataset_to_train():
    global model
    dataset = Datapreprocessor.readfile(openfile())
    dataset = [data.strip().split(' ') for data in dataset]
    dataset = Datapreprocessor.text_to_numlist(dataset)
    #train4dAll
    if len(dataset[0]) == 4:
        model = MultilayerPerceptron(5,5,3,1)
        train_x, train_y = Datapreprocessor.feature_label_split(dataset)
        model.train(np.array(train_x), np.array(train_y), 0.001, 1000)
    elif len(dataset[0]) == 6:
        model = MultilayerPerceptron(5,5,5,1)
        train_x, train_y = Datapreprocessor.feature_label_split(dataset)
        model.train(np.array(train_x), np.array(train_y), 0.001, 1000)


window = tk.Tk()
window.title('self driving car')
window.geometry('500x500')
canvas = tk.Canvas(window, bg='white', height=400, width=400).pack()
btn_track = tk.Button(window, text='select track data', command=draw_track).pack()
btn_train = tk.Button(window, text='select training data', command=select_dataset_to_train).pack()


window.mainloop()