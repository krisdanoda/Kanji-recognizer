from PIL import ImageGrab
import tkinter as tk
from tkinter.filedialog import asksaveasfile


def open_drawing_canvas():

    path = "input_images/Canvas/"
    lastx, lasty = 0, 0


    def xy(event):
        "Takes the coordinates of the mouse when you click the mouse"
        global lastx, lasty
        lastx, lasty = event.x, event.y


    def addLine(event):
        """Creates a line when you drag the mouse
        from the point where you clicked the mouse to where the mouse is now"""
        global lastx, lasty
        canvas.create_line((lastx, lasty, event.x, event.y), width=35, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        # this makes the new starting point of the drawing
        lastx, lasty = event.x, event.y


    def addWhiteLine(event):
        """Creates a line when you drag the mouse
        from the point where you clicked the mouse to where the mouse is now"""
        global lastx, lasty
        canvas.create_line((lastx, lasty, event.x, event.y), width=50, fill='grey', capstyle=tk.ROUND, smooth=tk.TRUE)
        # this makes the new starting point of the drawing
        lastx, lasty = event.x, event.y


    def save():
        print("Saving image...")
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        im = ImageGrab.grab((x, y, x1, y1))
        im.save(path + "kanji.png")
        root.destroy()



    root = tk.Tk()
    root.title("Draw a Kanji character")

    root.columnconfigure(0, weight=5)
    root.rowconfigure(0, weight=5)
    root.configure(background='white')

    canvas = tk.Canvas(root, bg="white", width=800, height=800)
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.bind("<Button-1>", xy)
    canvas.bind("<B1-Motion>", addLine)

    btn = tk.Button(root, text="Test", command=save)
    btn.pack(side=tk.BOTTOM)
    root.mainloop()


