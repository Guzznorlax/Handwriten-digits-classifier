import os
import sys
import io
import torch
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tkinter as tk
import PIL
from PIL import Image
import PIL.ImageOps
from alexnet import AlexNet
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
matplotlib.use("TkAgg")


class main:
    def __init__(self, master, model):
        self.master = master
        self.model = model
        self.res = ""
        self.pre = [None, None]
        self.bs = 8.5
        self.c = tk.Canvas(self.master, bd=3, relief="ridge", width=400, height=400, bg='white')
        self.c.pack(side=tk.LEFT, anchor="nw")

        f1 = tk.Frame(self.master, padx=5, pady=5)
        tk.Label(f1, text="Alexnet MNIST", fg="green", font=("", 40, "bold")).pack(pady=3)
        self.pr = tk.Label(f1, text="Prediction: None", fg="blue", font=("", 40, "bold"))
        self.pos = tk.Label(f1, text="Possibility: None", fg="blue", font=("", 40, "bold"))
        self.pr.pack(pady=3)
        self.pos.pack(pady=3)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.graph = self.fig.add_subplot(111)
        self.graph.set_ylim([0, 1.05])
        self.graph.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.plt_canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.plt_canvas.draw()
        self.plt_canvas.get_tk_widget().pack(side=tk.TOP, anchor="nw")
        self.toolbar = NavigationToolbar2Tk(self.plt_canvas, root)
        self.toolbar.update()
        self.plt_canvas.get_tk_widget().pack(side=tk.TOP, anchor="nw")

        tk.Button(f1, font=("", 15), fg="white", bg="red", text="Clear Canvas", command=self.clear).pack(side=tk.BOTTOM)

        f1.pack(side=tk.RIGHT, fill=tk.Y)
        self.c.bind("<Button-1>", self.putPoint)
        self.c.bind("<ButtonRelease-1>", self.getResult)
        self.c.bind("<B1-Motion>", self.paint)

    def getResult(self, e):
        ps = self.c.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = PIL.ImageOps.invert(img)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img_path = os.path.join(sys.path[0], "dist.png")
        img.save(img_path)

        pos_data, pos, self.res = predict(self.model, img_path)

        self.graph.bar(pos_data[0], pos_data[1])
        self.plt_canvas.draw()

        self.pr['text'] = "Prediction: " + str(self.res)
        self.pos['text'] = "Possibility: " + str(round(pos * 100, 2)) + "%"

    def clear(self):
        self.c.delete('all')
        self.graph.clear()
        self.graph.set_ylim([0, 1.05])
        self.graph.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.plt_canvas.draw_idle()

    def putPoint(self, e):
        self.c.create_oval(e.x - self.bs, e.y - self.bs, e.x + self.bs, e.y + self.bs, outline='black', fill='black')
        self.pre = [e.x, e.y]

    def paint(self, e):
        self.c.create_line(self.pre[0], self.pre[1], e.x, e.y, width=self.bs * 2, fill='black', capstyle=tk.ROUND,
                           smooth=True)

        self.pre = [e.x, e.y]


def load_model(model_path):
    """Loads a saved model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)

    return model


def preprocess_img(img_path):
    """Loads image with size=28 * 28, then converts it into nn readable tensor"""
    img = Image.open(img_path)
    img = img.convert('1')
    img = TF.to_tensor(img)
    img.unsqueeze_(0)

    return img


def predict(model, img_path):
    """Predicts a image."""
    img = preprocess_img(img_path)
    result = model(img)

    r_softmax = (F.softmax(result.data, 1))

    possibility, prediction = torch.max(r_softmax, 1)

    print("[INFO] Prediction:", prediction)

    r_softmax = r_softmax.tolist()[0]
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print("[RESULT] Prediction:", prediction.item())
    print("[RESULT] Possibility:", str(possibility.item() * 100) + "%")

    return [x, r_softmax], possibility.item(), prediction.item()


def imshow(tensor, title=None):
    """Converts a tensor to a PIL image."""
    unloader = torchvision.transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    model_path = os.path.join(sys.path[0], "MNIST_adam_v6_9914.pkl")
    model = load_model(model_path)

    root = tk.Tk()
    main(root, model)
    root.title('MNIST Digit Classifier')
    root.resizable(0, 0)
    root.mainloop()
