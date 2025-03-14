"""
autor: Saf
gitnub: https://github.com/DefessusSaf

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        self.v_scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # When changing the size of the internal frame, we update the scroll area
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

def parse_excel_range(range_str):
    parts = range_str.split(":")
    if len(parts) == 1:
        parts.append(parts[0])
    elif len(parts) != 2:
        raise ValueError(f"Неправильный формат диапазона: {range_str}")
    
    def cell_to_coords(cell):
        col = ord(cell[0].upper()) - ord("A") + 1
        row = int(cell[1:])
        return col, row

    start_col, start_row = cell_to_coords(parts[0])
    end_col, end_row = cell_to_coords(parts[1])
    
    if start_row > end_row or start_col > end_col:
        raise ValueError(f"Диапазон некорректен: {range_str}")
    
    return start_col, start_row, end_row

def letter_to_index(letter):
    letter = letter.strip().upper()
    result = 0
    for char in letter:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result

def load_data(file_path, time_range, thrust_range, pressure_range=None):
    df = pd.read_excel(file_path)
    
    time_col, time_start, time_end = parse_excel_range(time_range)
    thrust_col, thrust_start, thrust_end = parse_excel_range(thrust_range)
    
    time = pd.to_numeric(df.iloc[time_start - 1:time_end, time_col - 1], errors="coerce")
    thrust = pd.to_numeric(df.iloc[thrust_start - 1:thrust_end, thrust_col - 1], errors="coerce")
    
    if pressure_range:
        pressure_col, pressure_start, pressure_end = parse_excel_range(pressure_range)
        pressure = pd.to_numeric(df.iloc[pressure_start - 1:pressure_end, pressure_col - 1], errors="coerce")
        mask = ~np.isnan(time) & ~np.isnan(thrust) & ~np.isnan(pressure)
        return time[mask].values, thrust[mask].values, pressure[mask].values
    else:
        mask = ~np.isnan(time) & ~np.isnan(thrust)
        return time[mask].values, thrust[mask].values, None

def load_data_auto(file_path, start_row, end_row_input, start_col_letter):
    df = pd.read_excel(file_path, header=None)
    start_col_index = letter_to_index(start_col_letter)
    
    if not end_row_input.strip():
        end_row = df.shape[0]
    else:
        end_row = int(end_row_input)
    
    time = pd.to_numeric(df.iloc[start_row - 1:end_row, start_col_index - 1], errors="coerce")
    pressure = pd.to_numeric(df.iloc[start_row - 1:end_row, start_col_index], errors="coerce")
    thrust = pd.to_numeric(df.iloc[start_row - 1:end_row, start_col_index + 1], errors="coerce")
    
    mask = ~np.isnan(time) & ~np.isnan(pressure) & ~np.isnan(thrust)
    return time[mask].values, pressure[mask].values, thrust[mask].values

def calculate_total_impulse(time, thrust):
    return np.trapz(thrust, time)

def estimate_pressure(thrust, P_max, F_max):
    return P_max * (thrust / F_max) ** 0.8

def calculate_additional(time, thrust):
    if thrust.size == 0:
        return None, None
    max_thrust = np.max(thrust)
    idx_max = np.argmax(thrust)
    time_max = time[idx_max]
    return max_thrust, time_max

def plot_thrust_and_pressure(time, thrust, pressure, total_impulse, avg_pressure, save_dir=None, base_filename=""):
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    total_impulse_ns = total_impulse * 9.80665
    avg_pressure_pa = avg_pressure * 1e5 if avg_pressure is not None else None

    # We create one figure with two subscriptions
    fig, (ax_thrust, ax_pressure) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# ---The grave schedule ---    
    ax_thrust.plot(time, thrust, "b-", linewidth=2)
    ax_thrust.axvline(x=time[0], color="r", linestyle="--", linewidth=1.0)
    ax_thrust.axvline(x=time[-1], color="r", linestyle="--", linewidth=1.0)
    ax_thrust.axvspan(time[0], time[-1], facecolor="lightblue", alpha=0.3)
    ax_thrust.set_title("Тяга РДТТ", fontsize=16, fontweight="bold")
    ax_thrust.set_xlabel("Время, с", fontsize=12)
    ax_thrust.set_ylabel("Тяга, кг", fontsize=12)
    ax_thrust.grid(True, linestyle=":", alpha=0.7)
    ax_thrust.text(0.98, 0.95,
                   f"Полный импульс:\n{total_impulse:.2f} кг·с\n{total_impulse_ns:.2f} Н·с",
                   transform=ax_thrust.transAxes,
                   fontsize=10,
                   verticalalignment="top",
                   horizontalalignment="right",
                   bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.5"))

 # ---Pressure schedule ---
    if pressure is not None:
        ax_pressure.plot(time, pressure, "r-", linewidth=2)
        ax_pressure.axvline(x=time[0], color="b", linestyle="--", linewidth=1.0)
        ax_pressure.axvline(x=time[-1], color="b", linestyle="--", linewidth=1.0)
        ax_pressure.axvspan(time[0], time[-1], facecolor="lightblue", alpha=0.3)
        ax_pressure.set_title("Давление в камере", fontsize=16, fontweight="bold")
        ax_pressure.set_xlabel("Время, с", fontsize=12)
        ax_pressure.set_ylabel("Давление, бар", fontsize=12)
        ax_pressure.grid(True, linestyle=":", alpha=0.7)
        if avg_pressure is not None:
            ax_pressure.text(0.98, 0.95,
                            f"Среднее давление:\n{avg_pressure:.2f} бар\n{avg_pressure_pa:.2f} Па",
                            transform=ax_pressure.transAxes,
                            fontsize=10,
                            verticalalignment="top",
                            horizontalalignment="right",
                            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.5"))
    plt.tight_layout()

    if save_dir:
        import os
        save_path = os.path.join(save_dir, f"{base_filename}_thrust_pressure.png")
        plt.savefig(save_path)

    plt.show()
    plt.close(fig)

class ThrustApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Анализ Тяги РДТТ")
        
        # Variables for settings
        self.input_mode = tk.StringVar(value="file")   # One file or folder
        self.range_mode = tk.StringVar(value="manual")   # Manual input or autos

        # Create a scrolling container for the entire interface
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        scroll_frame = ScrollableFrame(container)
        scroll_frame.pack(fill="both", expand=True)
        self.main_frame = scroll_frame.scrollable_frame
        
        # ---------- The upper part of the interface: selection of settings ----------
        frame_source = tk.LabelFrame(self.main_frame, text="Источник данных")
        frame_source.pack(fill="x", padx=5, pady=5)
        tk.Radiobutton(frame_source, text="Один файл",
                       variable=self.input_mode, value="file").pack(side="left", padx=5, pady=5)
        tk.Radiobutton(frame_source, text="Папка с файлами",
                       variable=self.input_mode, value="folder").pack(side="left", padx=5, pady=5)
        
        frame_range = tk.LabelFrame(self.main_frame, text="Режим выбора диапазона")
        frame_range.pack(fill="x", padx=5, pady=5)
        tk.Radiobutton(frame_range, text="Ручной ввод",
                       variable=self.range_mode, value="manual").pack(side="left", padx=5, pady=5)
        tk.Radiobutton(frame_range, text="Автопоиск",
                       variable=self.range_mode, value="auto").pack(side="left", padx=5, pady=5)
        
        frame_file = tk.Frame(self.main_frame)
        frame_file.pack(fill="x", padx=5, pady=5)
        tk.Label(frame_file, text="Файл/Папка: ").pack(side="left", padx=5)
        self.file_entry = tk.Entry(frame_file, width=50)
        self.file_entry.pack(side="left", padx=5)
        self.browse_button = tk.Button(frame_file, text="Обзор", command=self.browse)
        self.browse_button.pack(side="left", padx=5)
        
        frame_output = tk.Frame(self.main_frame)
        frame_output.pack(fill="x", padx=5, pady=5)
        tk.Label(frame_output, text="Сохранить графики в (опционально): ").pack(side="left", padx=5)
        self.output_folder_entry = tk.Entry(frame_output, width=50)
        self.output_folder_entry.pack(side="left", padx=5)
        self.output_browse_button = tk.Button(frame_output, text="Обзор", command=self.browse_output)
        self.output_browse_button.pack(side="left", padx=5)
        
        # Frame of manual input of ranges
        self.frame_manual = tk.LabelFrame(self.main_frame, text="Ручной ввод диапазонов (формат: A1:A100)")
        self.frame_manual.pack(fill="x", padx=5, pady=5)
        tk.Label(self.frame_manual, text="Время: ").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.time_range_entry = tk.Entry(self.frame_manual, width=15)
        self.time_range_entry.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.frame_manual, text="Тяга: ").grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.thrust_range_entry = tk.Entry(self.frame_manual, width=15)
        self.thrust_range_entry.grid(row=0, column=3, padx=5, pady=2)
        tk.Label(self.frame_manual, text="Давление: ").grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.pressure_range_entry = tk.Entry(self.frame_manual, width=15)
        self.pressure_range_entry.grid(row=0, column=5, padx=5, pady=2)
        
        # Frame Auto -Sopos Range
        self.frame_auto = tk.LabelFrame(self.main_frame, text="Автопоиск диапазона")
        self.frame_auto.pack(fill="x", padx=5, pady=5)
        tk.Label(self.frame_auto, text="Номер начальной строки: ").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.start_row_entry = tk.Entry(self.frame_auto, width=10)
        self.start_row_entry.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.frame_auto, text="Номер конечной строки (пусто = до конца): ").grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.end_row_entry = tk.Entry(self.frame_auto, width=10)
        self.end_row_entry.grid(row=0, column=3, padx=5, pady=2)
        tk.Label(self.frame_auto, text="Буква первого столбца: ").grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.start_col_entry = tk.Entry(self.frame_auto, width=5)
        self.start_col_entry.grid(row=0, column=5, padx=5, pady=2)
        
        # Frame for entering fuel mass
        frame_fuel = tk.Frame(self.main_frame)
        frame_fuel.pack(fill="x", padx=5, pady=5)
        tk.Label(frame_fuel, text="Масса топлива (кг, опционально): ").pack(side="left", padx=5)
        self.fuel_mass_entry = tk.Entry(frame_fuel, width=10)
        self.fuel_mass_entry.pack(side="left", padx=5)
        
        #Frame with buttons
        frame_buttons = tk.Frame(self.main_frame)
        frame_buttons.pack(fill="x", padx=5, pady=5)
        self.process_button = tk.Button(frame_buttons, text="Запустить обработку", command=self.process)
        self.process_button.pack(side="left", padx=5)
        self.copy_button = tk.Button(frame_buttons, text="Копировать результаты", command=self.copy_results)
        self.copy_button.pack(side="left", padx=5)
        
        # ---------- Lower part: SCROLLLEDTEXT window for output of results ----------
        self.results_text = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Add binding for Ctrl + C (copying dedicated)
        self.results_text.bind("<Control-c>", self.copy_selection)

        self.range_mode.trace("w", self.toggle_range_frames)
        self.toggle_range_frames()

    def toggle_range_frames(self, *args):
        mode = self.range_mode.get()
        if mode == "manual":
            self.frame_manual.pack(fill="x", padx=5, pady=5)
            self.frame_auto.forget()
        else:
            self.frame_auto.pack(fill="x", padx=5, pady=5)
            self.frame_manual.forget()
            
    def browse(self):
        if self.input_mode.get() == "file":
            file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
            if file_path:
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, file_path)
        else:
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, folder_path)
                
    def browse_output(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_path)

    def copy_results(self):
        text = self.results_text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Готово", "Результаты скопированы в буфер обмена.")

    def copy_selection(self, event=None):
        """Копирует только выделенный фрагмент из ScrolledText."""
        try:
            selected_text = self.results_text.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except tk.TclError:
            # Nothing is highlighted
            pass
        return "break"

    def process(self):
        self.results_text.delete("1.0", tk.END)

        fuel_mass_str = self.fuel_mass_entry.get().strip()
        if fuel_mass_str:
            try:
                fuel_mass = float(fuel_mass_str)
            except ValueError:
                messagebox.showerror("Ошибка", "Неверное значение массы топлива.")
                return
        else:
            fuel_mass = None
        
        mode_source = self.input_mode.get()
        range_mode = self.range_mode.get()
        results = []
        errors = []
        output_folder = self.output_folder_entry.get().strip() or None
        
        if mode_source == "file":
            file_path = self.file_entry.get().strip()
            if not file_path:
                messagebox.showerror("Ошибка", "Не выбран файл.")
                return
            try:
                if range_mode == "manual":
                    time_range = self.time_range_entry.get().strip()
                    thrust_range = self.thrust_range_entry.get().strip()
                    pressure_range = self.pressure_range_entry.get().strip()
                    if pressure_range == "":
                        pressure_range = None
                    time, thrust, pressure = load_data(file_path, time_range, thrust_range, pressure_range)
                else:
                    start_row = int(self.start_row_entry.get().strip())
                    end_row_input = self.end_row_entry.get().strip()
                    start_col = self.start_col_entry.get().strip()
                    time, pressure, thrust = load_data_auto(file_path, start_row, end_row_input, start_col)
                    
                total_impulse = calculate_total_impulse(time, thrust)
                total_impulse_ns = total_impulse * 9.80665
                if fuel_mass is not None:
                    specific_impulse = total_impulse_ns / fuel_mass
                else:
                    specific_impulse = None
                avg_pressure = np.mean(pressure) if (pressure is not None and pressure.size > 0) else None

                max_thrust, time_of_max = calculate_additional(time, thrust)
                
                results.append((file_path, total_impulse, total_impulse_ns, avg_pressure, specific_impulse,
                                max_thrust, time_of_max))
                
                if output_folder:
                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    plot_thrust_and_pressure(time, thrust, pressure, total_impulse, avg_pressure,
                                             save_dir=output_folder, base_filename=base_filename)
            except Exception as e:
                errors.append((file_path, str(e)))
                
        else:
            folder_path = self.file_entry.get().strip()
            if not folder_path:
                messagebox.showerror("Ошибка", "Не выбрана папка.")
                return
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".xlsx")]
            if not files:
                messagebox.showerror("Ошибка", "В выбранной папке не найдено файлов .xlsx")
                return
            
            for file in files:
                try:
                    if range_mode == "manual":
                        time_range = self.time_range_entry.get().strip()
                        thrust_range = self.thrust_range_entry.get().strip()
                        pressure_range = self.pressure_range_entry.get().strip()
                        if pressure_range == "":
                            pressure_range = None
                        time, thrust, pressure = load_data(file, time_range, thrust_range, pressure_range)
                    else:
                        start_row = int(self.start_row_entry.get().strip())
                        end_row_input = self.end_row_entry.get().strip()
                        start_col = self.start_col_entry.get().strip()
                        time, pressure, thrust = load_data_auto(file, start_row, end_row_input, start_col)
                        
                    total_impulse = calculate_total_impulse(time, thrust)
                    total_impulse_ns = total_impulse * 9.80665
                    if fuel_mass is not None:
                        specific_impulse = total_impulse_ns / fuel_mass
                    else:
                        specific_impulse = None
                    avg_pressure = np.mean(pressure) if (pressure is not None and pressure.size > 0) else None

                    max_thrust, time_of_max = calculate_additional(time, thrust)
                    
                    results.append((file, total_impulse, total_impulse_ns, avg_pressure, specific_impulse,
                                    max_thrust, time_of_max))
                    
                    if output_folder:
                        base_filename = os.path.splitext(os.path.basename(file))[0]
                        plot_thrust_and_pressure(time, thrust, pressure, total_impulse, avg_pressure,
                                                 save_dir=output_folder, base_filename=base_filename)
                except Exception as e:
                    errors.append((file, str(e)))
        
        output_text = ""
        for res in results:
            file_label, ti, ti_ns, avg_p, si, max_t, t_max = res
            output_text += f"Файл: {file_label}\n"
            output_text += f"  Полный импульс: {ti:.2f} кг·с, {ti_ns:.2f} Н·с\n"
            if avg_p is not None:
                output_text += f"  Среднее давление: {avg_p:.2f} бар, {avg_p * 1e5:.2f} Па\n"
            if si is not None:
                output_text += f"  Удельный импульс: {si:.2f} м/с\n"
            if max_t is not None and t_max is not None:
                output_text += f"  Максимальная тяга: {max_t:.2f} кг, достигается в {t_max:.2f} с\n"
            output_text += "\n"
        
        self.results_text.insert(tk.END, output_text)
        
        if errors:
            error_messages = "\n".join([f"{fname}: {err}" for fname, err in errors])
            messagebox.showerror("Ошибки обработки", f"Некоторые файлы не удалось обработать:\n{error_messages}")


if __name__ == "__main__":
    try:
        app = ThrustApp()
        app.mainloop()
    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        messagebox.showerror("Ошибка", f"Произошла ошибка:\n{error_message}")
        raise
