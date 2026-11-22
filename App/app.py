import tkinter as tk
from App.interface import create_interface


def main():
    # Initialize the application
    root = tk.Tk()

    # Configure the main window
    root.title("ML Recycling Sorter")
    root.geometry("900x600")
    root.minsize(700, 500)

    # Load the user interface
    create_interface(root)

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()