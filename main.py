import tkinter as tk
from tkinter import filedialog
from DataPreprocessor import Datapreprocessor
from MLPerceptron import *
from simple_playground import *
import time
import random

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
            canvas.create_line(x0, y0, x1, y1)
        else:
            x0, y0 = coordinate_data[idx][0] + 100, -coordinate_data[idx][1] + 100
            x1, y1 = coordinate_data[0][0] + 100, -coordinate_data[0][1] + 100
            canvas.create_line(x0, y0, x1, y1)
model = None
feature_len = "None"
def select_dataset_to_train():
    global model, feature_len
    dataset = Datapreprocessor.readfile(openfile())
    dataset = [data.strip().split(' ') for data in dataset]
    dataset = Datapreprocessor.text_to_numlist(dataset)
    train_x, train_y = Datapreprocessor.feature_label_split(dataset)
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_y = Datapreprocessor.normalize_1d_np(train_y)
    train = []
    for record, target in zip(train_x, train_y):
        train.append((record, np.array(target,ndmin=1)))
    np.random.shuffle(train)
    #train4dAll
    feature_len = len(dataset[0])
    if len(dataset[0]) == 4:
        model = NeuralNetwork([
            InputLayer(inputs=3),
            SigmoidLayer(inputs=3, outputs=25),
            SigmoidLayer(inputs=25, outputs=25),
            SigmoidLayer(inputs=25, outputs=25),
            SigmoidLayer(inputs=25, outputs=1),
        ])
    elif len(dataset[0]) == 6:
        model = NeuralNetwork([
            InputLayer(inputs=5),
            SigmoidLayer(inputs=5, outputs=25),
            SigmoidLayer(inputs=25, outputs=25),
            SigmoidLayer(inputs=25, outputs=25),
            SigmoidLayer(inputs=25, outputs=1),
        ])
    model.fit(train, epochs=int(entry_epoch.get()), learning_rate=float(entry_learning_rate.get()))


def print_result():
    global model
    position_list, state_list, action_list = run_example(model, feature_len)
    animate_ball(window, canvas, position_list, state_list)
    if feature_len == 4:
        save_4d_result(state_list, action_list)
    elif feature_len == 6:
        save_6d_result(position_list, state_list, action_list)

def animate_ball(Window, canvas, position_list, state_list):
    pos_x, pos_y = position_list[0].x, -position_list[0].y
    pos_x, pos_y = float(pos_x), float(pos_y)
    ball = canvas.create_oval(pos_x+100-3,pos_y+100-3,pos_x+100+3,pos_y+100+3,fill="Red", outline="Black", width=4)
    for idx, position in enumerate(position_list[1:]):
        next_x, next_y = position.x, -position.y
        front, right, left = state_list[idx][0], state_list[idx][1], state_list[idx][2]
        next_x, next_y = float(next_x), float(next_y)
        xinc = next_x - pos_x
        yinc = next_y - pos_y
        pos_x, pos_y = next_x, next_y
        canvas.move(ball,xinc,yinc)
        pos_lbl.configure(text=f"x: {position.x}, y: {position.y}")
        distance_lbl.configure(text=f"front: {front}, right: {right}, left: {left}")
        Window.update()
        time.sleep(0.2)

def save_4d_result(state_list, action_list):
    file = open("4d_result.txt", "w")
    for i in range(len(action_list)):
        str = ""
        file.write(f"{state_list[i]} {action_list[i][0]-40}\n")
    file.close

def save_6d_result(position_list, state_list, action_list):
    file = open("6d_result.txt", "w")
    for i in range(len(action_list)):
        file.write(f"{position_list[i]} {state_list[i]} {action_list[i][0]-40}\n")
    file.close()

window = tk.Tk()
window.title('self driving car')
window.geometry('500x500')
canvas = tk.Canvas(window, bg='white', height=200, width=200)
canvas.pack()


tk.Label(window, text='epoch: ').pack()
epoch = tk.IntVar()
entry_epoch = tk.Entry(window, textvariable=epoch)
entry_epoch.pack()
tk.Label(window, text='learning rate: ').pack()
learning_rate = tk.DoubleVar()
entry_learning_rate = tk.Entry(window, textvariable=learning_rate)
entry_learning_rate.pack()

pos_lbl = tk.Label(window, text='car position')
pos_lbl.pack()

distance_lbl = tk.Label(window, text='front & right & left distance')
distance_lbl.pack()


btn_track = tk.Button(window, text='select track data', command=draw_track)
btn_track.pack()
btn_train = tk.Button(window, text='select training data', command=select_dataset_to_train)
btn_train.pack()
btn_result = tk.Button(window, text='print_result', command=print_result)
btn_result.pack()

window.mainloop()