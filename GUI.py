import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox
from PIL import ImageTk
import gc
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms, models

configuration_models = [
    {
        "learning_rate": 1e-5,
        "batch_size": 32,
        "picture_size": [512, 512],
        "weight_decay": 0,
        "max_epochs": 15,
        "shuffle_test": True
    },
    {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "picture_size": [256, 256],
        "weight_decay": 0,
        "max_epochs": 15,
        "shuffle_test": True
    }
]
choice = ("Predict/Detect DR's scale", "Detect the presence of diabetic retinopathy")
models_paths = ["model/convnext_small(picture_size=512x512).pth",
                "model/convnext_small(Best_params-15-without_wd-256-healthy).pth"]
current_path = ""

class DiabeticRetinopathyModel(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 batch_size: int,
                 picture_size: [int, int],
                 weight_decay: float,
                 max_epochs: int,
                 shuffle_test: bool):
        super(DiabeticRetinopathyModel, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.picture_size = picture_size
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.shuffle_test = shuffle_test

        self.convnext_small = models.convnext_small(pretrained=True)

        self.accuracy_train = Accuracy(compute_on_step=False)
        self.accuracy_valid = Accuracy(compute_on_step=False)
        self.accuracy_test = Accuracy(compute_on_step=False)

    def forward(self, x):
        return self.convnext_small.forward(x)

    def training_step(self, batch, batch_nb):
        x, y = batch

        logits = self.convnext_small(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log('train/val_loss', loss, prog_bar=True)
        self.accuracy_train.update(preds, y)
        return loss

    def training_epoch_end(self, outputs):
        avg = self.accuracy_train.compute()
        self.log('train/val_acc', avg, on_step=False, on_epoch=True, prog_bar=True)
        gc.collect()

    def validation_step(self, batch, batch_nb):
        x, y = batch

        logits = self.convnext_small(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log('validation/val_loss', loss, prog_bar=True)
        self.accuracy_valid.update(preds, y)
        return loss

    def validation_epoch_end(self, outputs):
        avg = self.accuracy_valid.compute()
        self.log('validation/val_acc', avg, on_step=False, on_epoch=True, prog_bar=True)
        gc.collect()

    def test_step(self, batch, batch_nb):
        x, y = batch

        logits = self.convnext_small(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log('test/val_loss', loss, prog_bar=True)
        self.accuracy_test.update(preds, y)
        return loss

    def test_epoch_end(self, outputs):
        avg = self.accuracy_test.compute()
        self.log('test/val_acc', avg, on_step=False, on_epoch=True, prog_bar=True)
        gc.collect()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


config, model = None, None


def load_model():
    global model, config
    config = configuration_models[choice.index(combo.get())]
    model = DiabeticRetinopathyModel(config["learning_rate"],
                                     config["batch_size"],
                                     config["picture_size"],
                                     config["weight_decay"],
                                     config["max_epochs"],
                                     config["shuffle_test"])


if __name__ == "__main__":
    # --------------------------------------------------------- Model
    gc.collect()
    pl.seed_everything(7)

    # --------------------------------------------------------- GUI
    window = tk.Tk()
    window.title("Diabetic Retinopathy Detection")

    window.rowconfigure(0, minsize=700, weight=1)
    window.columnconfigure(1, minsize=700, weight=1)
    txt_edit = tk.Text(window, state="normal", relief=tk.RAISED)
    txt_edit.grid(row=1, column=1, sticky="nsew")

    management_model = tk.Frame(window, relief=tk.RIDGE, bd=3, width=400, height=200)
    management_model.grid(row=0, column=0, sticky="ns")


    def open_file():
        global current_path
        """Open a file for editing."""
        filepath = askopenfilename(
            title="Choose an image",
            filetypes=[("Image Files", ("*.png", "*.jpeg", "*.jpg"))]
        )

        if not filepath:
            txt_edit.insert(tk.END, "The image is not found, format should be `*.png`, `*.jpeg`, `*.jpg`.\n")
            return

        current_path = filepath
        im = Image.open(filepath)
        resized_image = im.resize((720, 480), Image.Resampling.LANCZOS)
        tkimage = ImageTk.PhotoImage(resized_image)

        image_label = tk.Label(window, width=200, height=20, image=tkimage)
        image_label.grid(row=0, column=1, sticky="nsew")
        image_label.image = tkimage


    btn_open = tk.Button(management_model, text="Open", command=open_file, height=3)


    def detect():
        global model, models_paths, config, choice, current_path
        if current_path == "":
            txt_edit.insert(tk.END, "Choose the image first.\n")
            return

        txt_edit.insert(tk.END, "Detecting the image using method `" + combo.get() + "`...\n")
        print(models_paths[choice.index(combo.get())])
        model.load_state_dict(torch.load(models_paths[choice.index(combo.get())]))
        model.eval()
        model.parameters()

        current_image = Image.open(current_path)

        # print(models_paths[choice.index(combo.get())], config["picture_size"][0], config["picture_size"][1])
        transform = transforms.Compose([
            transforms.Resize((config["picture_size"][0], config["picture_size"][1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(current_image)
        prediction = model(image.unsqueeze(0))
        scale = torch.argmax(prediction, dim=1)
        # print(current_path, prediction.shape, torch.argmax(prediction, dim=1))
        # print(scale, scale.numpy()[0], scale.item(), str(scale.item()))

        if choice.index(combo.get()) == 0:
            txt_edit.insert(tk.END, "Image: " + current_path + "; Scale: " + str(scale.item()) + ".\n")
        else:
            txt_edit.insert(tk.END, "Image: " + current_path + "; Healthy: " + str(scale.item() == 0) + ".\n")


    btn_detect = tk.Button(management_model, text="Detect", command=detect, height=3)

    btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    btn_detect.place(relwidth=0.95, rely=0.5, relx=0.5, x=0, y=0, anchor=tk.CENTER)


    # btn_detect.grid(row=2, column=0, sticky="news", padx=5, pady=5)

    def callbackFunc(event):
        c = event.widget.get()
        load_model()
        print(c)


    n = tk.StringVar()
    combo = Combobox(management_model, state="readonly", textvariable=n, width=40)

    combo['values'] = choice
    combo.current(0)
    combo.grid(column=0, row=1)
    combo.bind("<<ComboboxSelected>>", callbackFunc)

    load_model()
    window.mainloop()
