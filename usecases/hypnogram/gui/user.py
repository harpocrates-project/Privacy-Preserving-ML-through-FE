import tkinter as tk
from tkinter import filedialog

from base import CommandLineWrapper
from utils import plot_hypnogram

class UserWrapper(CommandLineWrapper):
    def __init__(self, master=None, default_exe=None, default_analyst=None, default_server=None):
        self.server = tk.StringVar(value=default_server)
        self.id = tk.IntVar()
        self.hypnogram_data = tk.StringVar()
        super().__init__(master, default_exe)
        self.hypnogram_data.trace_add("write", self.on_add_hypnogram_data)

    def on_add_hypnogram_data(self, *args, **kwargs):
        file_path = self.hypnogram_data.get()
        plot_hypnogram(file_path)

    @property
    def terminal_cmd(self):
        exe_path = self.executable_path.get().strip()
        server = self.server.get().strip()
        id = self.id.get()
        hypnogram = self.hypnogram_data.get().strip()

        cmd = exe_path
        cmd += f" {server} {id} {hypnogram}"

        return [part.format(cmd=cmd) for part in self.terminal_cmd_template]

    def browse_hypnogram_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Hypnogram Data file",
            filetypes=[("All files", "*")],
        )
        if file_path:
            self.hypnogram_data.set(file_path)

    def create_widgets(self):
        # Parameters frame (Server, Hypnogram Data)
        param_frame = tk.Frame(self)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Server
        tk.Label(param_frame, text="Server:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(param_frame, textvariable=self.server, width=15).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Number of Records (read‑only)
        tk.Label(param_frame, text="ID:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(
            param_frame,
            textvariable=self.id,
            width=10,
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Hypnogram Data
        tk.Label(param_frame, text="Hypnogram Data:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(param_frame, textvariable=self.hypnogram_data, width=25).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        tk.Button(
            param_frame, text="Browse...", command=self.browse_hypnogram_data
        ).pack(side=tk.LEFT, padx=5)



        # Add the standard executable/browse/run widgets
        super().create_widgets()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("User")
    app = UserWrapper(
        master=root,
        default_exe="./user",
        default_server="localhost:50505",
    )
    root.mainloop()
