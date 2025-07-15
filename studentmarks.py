import os
import io
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

# Globals
files = []
combined_df = pd.DataFrame()
subject_columns = []
predictions_df = pd.DataFrame()

def assign_grade(mark):
    if pd.isnull(mark): return '-'
    if mark >= 90: return 'A'
    elif mark >= 80: return 'B'
    elif mark >= 70: return 'C'
    elif mark >= 60: return 'D'
    elif mark >= 50: return 'E'
    else: return 'F'

def grades_from_marks_array(marks):
    return [assign_grade(m) for m in marks]

def browse_files():
    global combined_df, subject_columns, predictions_df
    file_paths = filedialog.askopenfilenames(title="Select up to 5 CSV files", filetypes=[("CSV files", "*.csv")])
    if len(file_paths) > 5:
        messagebox.showerror("Error", "Please select up to 5 files only.")
        return
    if not file_paths:
        return
    files.clear()
    files.extend(file_paths)
    file_label.config(text="\n".join(os.path.basename(f) for f in files))
    dfs = []
    for path in files:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            messagebox.showerror("File Read Error", f"Could not read file {os.path.basename(path)}:\n{e}")
            return
    if not dfs:
        messagebox.showerror("No Data", "No valid CSV files loaded.")
        return
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.insert(0, "Student_No", combined_df.index + 1)
    subject_columns.clear()
    subject_columns.extend([col for col in combined_df.columns if col not in ['Student_ID', 'Student_No'] and pd.api.types.is_numeric_dtype(combined_df[col])])
    predictions_df = pd.DataFrame()
    update_subject_menu()
    update_roll_menu()
    if subject_columns:
        display_data(subject_columns[0])
    update_stats_panel()
    status_label.config(text="Files loaded successfully!")

def update_subject_menu():
    subject_dropdown['values'] = subject_columns
    if subject_columns:
        subject_var.set(subject_columns[0])
    else:
        subject_var.set('')

def update_roll_menu():
    if 'Student_ID' in combined_df.columns:
        roll_values = combined_df['Student_ID'].dropna().unique().tolist()
        roll_dropdown['values'] = roll_values
        if roll_values:
            roll_var.set(roll_values[0])
        else:
            roll_var.set('')
    else:
        roll_dropdown['values'] = []
        roll_var.set('')

def display_data(subject):
    for widget in table_frame.winfo_children():
        widget.destroy()
    if combined_df.empty or not subject:
        return
    container = ttk.Frame(table_frame)
    container.pack(expand=True, fill='both')
    tree_scroll_y = ttk.Scrollbar(container, orient="vertical")
    tree_scroll_x = ttk.Scrollbar(container, orient="horizontal")
    tree = ttk.Treeview(container, columns=["Student_No", subject], show='headings',
                        yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
    tree.heading("Student_No", text="Student No")
    tree.heading(subject, text=subject)
    tree.column("Student_No", width=100, anchor="center")
    tree.column(subject, width=150, anchor="center")
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    tree_scroll_y.pack(side='right', fill='y')
    tree_scroll_x.pack(side='bottom', fill='x')
    tree.pack(expand=True, fill='both')
    for _, row in combined_df.iterrows():
        tree.insert('', 'end', values=(row["Student_No"], row[subject]))

def view_all_data():
    for widget in table_frame.winfo_children():
        widget.destroy()

    if combined_df.empty:
        return

    df = combined_df.copy()

    # Prepare columns: keep non-subject columns as-is, and for each subject, add [subject, subject_Grade]
    display_columns = []
    for col in df.columns:
        if col not in subject_columns:
            display_columns.append(col)
    for subject in subject_columns:
        df[f"{subject}_Grade"] = df[subject].apply(assign_grade)
        display_columns.append(subject)
        display_columns.append(f"{subject}_Grade")

    container = ttk.Frame(table_frame)
    container.pack(expand=True, fill='both')

    tree_scroll_y = ttk.Scrollbar(container, orient="vertical")
    tree_scroll_x = ttk.Scrollbar(container, orient="horizontal")

    tree = ttk.Treeview(container, columns=display_columns, show='headings',
                        yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    tree_scroll_y.pack(side='right', fill='y')
    tree_scroll_x.pack(side='bottom', fill='x')
    tree.pack(expand=True, fill='both')

    for col in display_columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    for _, row in df.iterrows():
        values = [row[col] for col in display_columns]
        tree.insert('', 'end', values=values)

    status_label.config(text="All student data with grades displayed.")

def plot_subject_data():
    selected_subject = subject_var.get()
    if not selected_subject or combined_df.empty:
        messagebox.showwarning("No Subject", "Please load CSV files and select a subject.")
        return
    if selected_subject not in combined_df.columns:
        messagebox.showerror("Invalid Subject", f"Subject '{selected_subject}' not found in data.")
        return
    for widget in table_frame.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(combined_df["Student_No"], combined_df[selected_subject], color='darkblue', s=12)
    ax.set_title(f"{selected_subject} vs Student_No")
    ax.set_xlabel("Student No")
    ax.set_ylabel(selected_subject)
    ax.grid(True)
    canvas_chart = FigureCanvasTkAgg(fig, master=table_frame)
    canvas_chart.draw()
    canvas_chart.get_tk_widget().pack(expand=True, fill='both')

def update_stats_panel():
    if combined_df.empty:
        stats_var.set("No data loaded.")
        return
    subject = subject_var.get() or (subject_columns[0] if subject_columns else None)
    if subject and subject in combined_df.columns:
        total_students = len(combined_df)
        mean = combined_df[subject].mean()
        minimum = combined_df[subject].min()
        maximum = combined_df[subject].max()
        stats_var.set(f"Students: {total_students} | {subject} → Mean: {mean:.2f}, Min: {minimum}, Max: {maximum}")
    else:
        stats_var.set("No data loaded.")

def predict_marks():
    global predictions_df
    if combined_df.empty or not subject_columns:
        messagebox.showwarning("No Data", "Please load data before prediction.")
        return
    predictions = combined_df[["Student_No"]].copy()
    for subject in subject_columns:
        X = combined_df[["Student_No"]].values
        y = combined_df[subject].values
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X)
        pred = np.clip(pred, 0, 100)
        predictions[f"{subject}_Predicted"] = pred.round(2)
    predictions_df = predictions.copy()
    display_prediction_results(predictions_df)

def display_prediction_results(pred_df):
    for widget in table_frame.winfo_children():
        widget.destroy()
    cols = pred_df.columns.tolist()
    container = ttk.Frame(table_frame)
    container.pack(expand=True, fill='both')
    tree_scroll_y = ttk.Scrollbar(container, orient="vertical")
    tree_scroll_x = ttk.Scrollbar(container, orient="horizontal")
    tree = ttk.Treeview(container, columns=cols, show='headings',
                        yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    tree_scroll_y.pack(side='right', fill='y')
    tree_scroll_x.pack(side='bottom', fill='x')
    tree.pack(side='left', fill='both', expand=True)
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor="center")
    for _, row in pred_df.iterrows():
        tree.insert('', 'end', values=list(row))
    status_label.config(text="Prediction completed and displayed.")

def display_metric_table(metric_name, results):
    for widget in table_frame.winfo_children():
        widget.destroy()
    container = ttk.Frame(table_frame)
    container.pack(expand=True, fill='both')
    tree_scroll_y = ttk.Scrollbar(container, orient="vertical")
    tree_scroll_x = ttk.Scrollbar(container, orient="horizontal")
    tree = ttk.Treeview(container, columns=["Subject", metric_name], show='headings',
                        yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    tree_scroll_y.pack(side='right', fill='y')
    tree_scroll_x.pack(side='bottom', fill='x')
    tree.pack(expand=True, fill='both')
    tree.heading("Subject", text="Subject")
    tree.heading(metric_name, text=metric_name)
    tree.column("Subject", width=200, anchor="center")
    tree.column(metric_name, width=150, anchor="center")
    for res in results:
        subject, value = res.split(": ")
        tree.insert('', 'end', values=(subject, value))
    status_label.config(text=f"{metric_name} results displayed.")

def calculate_metrics(metric_name):
    global predictions_df
    if combined_df.empty or predictions_df.empty:
        messagebox.showwarning("No Data", f"Load data and run prediction before calculating {metric_name}.")
        return
    results = []
    for subject in subject_columns:
        actual_marks = combined_df[subject].values
        predicted_marks = predictions_df[subject + "_Predicted"].values
        actual_grades = grades_from_marks_array(actual_marks)
        predicted_grades = grades_from_marks_array(predicted_marks)
        filtered_actual, filtered_pred = zip(*[(a, p) for a, p in zip(actual_grades, predicted_grades) if a != '-' and p != '-'])
        if not filtered_actual:
            results.append(f"{subject}: No valid data")
            continue
        if metric_name == "Precision":
            score = precision_score(filtered_actual, filtered_pred, average='macro', zero_division=0)
        elif metric_name == "Recall":
            score = recall_score(filtered_actual, filtered_pred, average='macro', zero_division=0)
        elif metric_name == "Accuracy":
            score = accuracy_score(filtered_actual, filtered_pred)
        else:
            score = None
        results.append(f"{subject}: {score:.4f}")
    display_metric_table(metric_name, results)

def display_student_marks():
    for widget in table_frame.winfo_children():
        widget.destroy()
    if combined_df.empty:
        messagebox.showwarning("No Data", "Please load data first.")
        return
    selected_roll = roll_var.get()
    if not selected_roll:
        messagebox.showwarning("No Selection", "Please select a Student Roll Number.")
        return
    student_row = combined_df[combined_df['Student_ID'] == selected_roll]
    if student_row.empty:
        messagebox.showerror("Not Found", f"No data found for Student ID: {selected_roll}")
        return
    data = student_row[subject_columns].iloc[0].to_dict()
    data_with_grades = {subj: f"{mark} ({assign_grade(mark)})" for subj, mark in data.items()}
    container = ttk.Frame(table_frame)
    container.pack(expand=True, fill='both')
    tree = ttk.Treeview(container, columns=["Subject", "Mark (Grade)"], show='headings')
    tree.heading("Subject", text="Subject")
    tree.heading("Mark (Grade)", text="Mark (Grade)")
    tree.column("Subject", width=200, anchor="center")
    tree.column("Mark (Grade)", width=150, anchor="center")
    for subj, mark_grade in data_with_grades.items():
        tree.insert('', 'end', values=(subj, mark_grade))
    tree.pack(expand=True, fill='both')
    status_label.config(text=f"Marks displayed for Student ID: {selected_roll}")

# GUI Setup
app = ttk.Window(title="Student Marks Analyzer", size=(1150, 780))
app.place_window_center()
style = ttk.Style()
stats_var = tk.StringVar(value="Welcome! Please load student data.")
subject_var = tk.StringVar()
roll_var = tk.StringVar()
current_theme = {"name": "flatly"}

def show_top_bottom_performers():
    for widget in table_frame.winfo_children():
        widget.destroy()
    if combined_df.empty or not subject_columns:
        messagebox.showwarning("No Data", "Please load data first.")
        return

    # Create scrollable container
    container = ttk.Frame(table_frame)
    container.pack(expand=True, fill='both')

    canvas_widget = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas_widget.yview)
    scrollable_frame = ttk.Frame(canvas_widget)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_widget.configure(scrollregion=canvas_widget.bbox("all"))
    )

    canvas_widget.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_widget.configure(yscrollcommand=scrollbar.set)

    canvas_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for subject in subject_columns:
        top10 = combined_df[['Student_ID', 'Student_No', subject]].nlargest(10, subject)
        bottom10 = combined_df[['Student_ID', 'Student_No', subject]].nsmallest(10, subject)

        # Section label
        ttk.Label(scrollable_frame, text=f"🏆 Top 10 Performers in {subject}",
                  font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        tree_top = ttk.Treeview(scrollable_frame, columns=["Student_ID", "Student_No", subject], show='headings', height=10)
        for col in ["Student_ID", "Student_No", subject]:
            tree_top.heading(col, text=col)
            tree_top.column(col, width=120, anchor="center")
        for _, row in top10.iterrows():
            tree_top.insert('', 'end', values=(row["Student_ID"], row["Student_No"], row[subject]))
        tree_top.pack(pady=5, padx=10, fill="x")

        ttk.Label(scrollable_frame, text=f"🔻 Bottom 10 Performers in {subject}",
                  font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        tree_bottom = ttk.Treeview(scrollable_frame, columns=["Student_ID", "Student_No", subject], show='headings', height=10)
        for col in ["Student_ID", "Student_No", subject]:
            tree_bottom.heading(col, text=col)
            tree_bottom.column(col, width=120, anchor="center")
        for _, row in bottom10.iterrows():
            tree_bottom.insert('', 'end', values=(row["Student_ID"], row["Student_No"], row[subject]))
        tree_bottom.pack(pady=5, padx=10, fill="x")

    status_label.config(text="Top and Bottom performers displayed.")

def export_graphs_to_pdf():
    if combined_df.empty or not subject_columns:
        messagebox.showwarning("No Data", "Please load CSV files before exporting.")
        return

    # Ask where to save the PDF
    pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        return

    # Create canvas
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 50

    for subject in subject_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(combined_df["Student_No"], combined_df[subject], color='blue', s=10)
        ax.set_title(f"{subject} vs Student_No")
        ax.set_xlabel("Student No")
        ax.set_ylabel(subject)
        ax.grid(True)

        # Save plot to a BytesIO object
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)

        # Insert image into PDF
        img = ImageReader(buf)
        c.drawImage(img, margin, height / 2 - 100, width=width - 2 * margin, preserveAspectRatio=True, mask='auto')
        c.showPage()

    c.save()
    messagebox.showinfo("Export Complete", f"PDF successfully saved to:\n{pdf_path}")
    
def edit_student_marks_gui():
    for widget in table_frame.winfo_children():
        widget.destroy()

    if combined_df.empty:
        messagebox.showwarning("No Data", "Please load data first.")
        return

    selected_roll = roll_var.get()
    if not selected_roll:
        messagebox.showwarning("No Selection", "Please select a Student Roll Number.")
        return

    student_row = combined_df[combined_df['Student_ID'] == selected_roll]
    if student_row.empty:
        messagebox.showerror("Not Found", f"No data found for Student ID: {selected_roll}")
        return

    ttk.Label(table_frame, text=f"Editing Marks for Student ID: {selected_roll}",
              font=("Helvetica", 13, "bold")).pack(pady=10)

    canvas_frame = ttk.Frame(table_frame)
    canvas_frame.pack(expand=True, fill='both', padx=10, pady=10)

    canvas_widget = tk.Canvas(canvas_frame)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas_widget.yview)
    scrollable_frame = ttk.Frame(canvas_widget)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_widget.configure(scrollregion=canvas_widget.bbox("all"))
    )

    canvas_widget.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_widget.configure(yscrollcommand=scrollbar.set)

    canvas_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    entries = {}

    for subject in subject_columns:
        ttk.Label(scrollable_frame, text=subject, font=("Helvetica", 10)).pack(pady=(5, 0))
        current_value = student_row[subject].values[0]
        entry = ttk.Entry(scrollable_frame, width=25)
        entry.insert(0, str(current_value))
        entry.pack(pady=(0, 5))
        entries[subject] = entry

    def save_edits():
        for subject, entry in entries.items():
            try:
                new_value = float(entry.get())
                combined_df.loc[combined_df['Student_ID'] == selected_roll, subject] = new_value
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter valid number for {subject}.")
                return
        status_label.config(text=f"✅ Marks updated for Student ID: {selected_roll}")
        update_stats_panel()
        display_student_marks()

    ttk.Button(table_frame, text="✅ Save Changes", command=save_edits, bootstyle="success").pack(pady=10)
    
def export_data_with_grades():
    if combined_df.empty:
        messagebox.showwarning("No Data", "Please load student data first.")
        return

    df = combined_df.copy()

    # Insert grades and reorder columns: [subject, subject_Grade] side-by-side
    for subject in subject_columns:
        df[f"{subject}_Grade"] = df[subject].apply(assign_grade)

    # Maintain logical column order
    base_columns = [col for col in df.columns if col not in subject_columns and not col.endswith('_Grade')]
    ordered_subjects = []
    for subject in subject_columns:
        ordered_subjects.append(subject)
        ordered_subjects.append(f"{subject}_Grade")

    final_columns = base_columns + ordered_subjects
    df = df[final_columns]

    # Ask user where to save
    export_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if export_path:
        df.to_csv(export_path, index=False)
        messagebox.showinfo("Exported", f"✅ Data exported with grades to:\n{export_path}")
        
def export_student_report():
    if combined_df.empty:
        messagebox.showwarning("No Data", "Please load data first.")
        return
    selected_roll = roll_var.get()
    if not selected_roll:
        messagebox.showwarning("No Selection", "Please select a Student Roll Number.")
        return

    student = combined_df[combined_df['Student_ID'] == selected_roll]
    if student.empty:
        messagebox.showerror("Not Found", f"No student found with ID {selected_roll}.")
        return

    # Ask where to save the report
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not save_path:
        return

    student_data = student.iloc[0]
    subject_marks = [student_data[subj] for subj in subject_columns]
    grades = [assign_grade(mark) for mark in subject_marks]

    # Plot marks chart
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(subject_columns, subject_marks, marker='o', color='navy')
    ax.set_title(f"Performance in Subjects")
    ax.set_ylabel("Marks")
    ax.set_ylim(0, 100)
    ax.grid(True)
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Create PDF
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    margin = 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - 50, "Student Performance Report")

    c.setFont("Helvetica", 12)
    c.drawString(margin, height - 80, f"Student ID: {selected_roll}")
    c.drawString(margin, height - 100, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Draw table of marks and grades
    y = height - 140
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Subject")
    c.drawString(margin + 200, y, "Marks")
    c.drawString(margin + 300, y, "Grade")
    c.setFont("Helvetica", 11)
    y -= 20
    for subj, mark, grade in zip(subject_columns, subject_marks, grades):
        c.drawString(margin, y, subj)
        c.drawString(margin + 200, y, str(mark))
        c.drawString(margin + 300, y, grade)
        y -= 20

    # Insert chart image
    image = ImageReader(buf)
    c.drawImage(image, margin, 80, width=width - 2 * margin, height=200, preserveAspectRatio=True)

    c.save()
    messagebox.showinfo("Success", f"PDF report saved to:\n{save_path}")
      
def toggle_theme():
    new_theme = "superhero" if current_theme["name"] == "flatly" else "flatly"
    current_theme["name"] = new_theme
    style.theme_use(new_theme)
    app.update_idletasks()
    status_label.config(text=f"Theme changed to {new_theme.title()}")

# Layout
stats_label = ttk.Label(app, textvariable=stats_var, font=("Helvetica", 11), anchor="center", bootstyle="info")
stats_label.pack(fill="x", padx=10, pady=(5, 0))

left_frame = ttk.Frame(app, padding=20, bootstyle="light")
left_frame.pack(side="left", fill="y", padx=10, pady=10)

table_frame = ttk.Frame(app, padding=10, bootstyle="secondary")
table_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# --- Title
ttk.Label(left_frame, text="📂 Student Marks Analyzer", font=("Helvetica", 18, "bold"), bootstyle="primary").pack(pady=10)

# --- Browse Files First
ttk.Button(left_frame, text="📂 Browse CSV Files", command=browse_files, bootstyle="success-outline").pack(pady=5)
file_label = ttk.Label(left_frame, text="No files selected.", wraplength=200, font=("Helvetica", 9))
file_label.pack(pady=5)

# --- Subject Selection
ttk.Label(left_frame, text="📊 Choose Subject:", font=("Helvetica", 10)).pack(pady=5)
subject_dropdown = ttk.Combobox(left_frame, textvariable=subject_var, state="readonly", width=25)
subject_dropdown.pack(pady=5)
subject_dropdown.bind("<<ComboboxSelected>>", lambda e: (display_data(subject_var.get()), update_stats_panel()))

# --- Plot & Prediction Buttons
ttk.Button(left_frame, text="📈 Plot Subject Marks", command=plot_subject_data, bootstyle="warning").pack(pady=5)
ttk.Button(left_frame, text="🔮 Predict Marks", command=predict_marks, bootstyle="info").pack(pady=5)
ttk.Button(left_frame, text="🔎 Precision", command=lambda: calculate_metrics("Precision"), bootstyle="success-outline").pack(pady=5)

ttk.Button(left_frame, text="🎯 Recall", command=lambda: calculate_metrics("Recall"), bootstyle="info-outline").pack(pady=5)
ttk.Button(left_frame, text="✅ Accuracy", command=lambda: calculate_metrics("Accuracy"), bootstyle="primary-outline").pack(pady=5)

# --- Roll Number Selection
ttk.Label(left_frame, text="🆔 Select Student Roll Number:", font=("Helvetica", 10)).pack(pady=5)
roll_dropdown = ttk.Combobox(left_frame, textvariable=roll_var, state="readonly", width=25)
roll_dropdown.pack(pady=5)
ttk.Button(left_frame, text="📜 View Student Marks", command=display_student_marks, bootstyle="warning-outline").pack(pady=5)

# --- Data & Edit Options
ttk.Button(left_frame, text="🗃️ View All Students Data", command=view_all_data, bootstyle="secondary").pack(pady=5)
ttk.Button(left_frame, text="✏️ Edit Student Marks", command=edit_student_marks_gui, bootstyle="success-outline").pack(pady=5)
ttk.Button(left_frame, text="🧾 Export Student PDF Report", command=export_student_report, bootstyle="danger").pack(pady=5)

# --- Export Last
ttk.Button(left_frame, text="🏅 Show Top & Bottom Performers", command=show_top_bottom_performers, bootstyle="info").pack(pady=5)
ttk.Button(left_frame, text="💾 Export CSV with Grades", command=export_data_with_grades, bootstyle="info-outline").pack(pady=3)
ttk.Button(left_frame, text="📄 Export PDF of All Graphs", command=export_graphs_to_pdf, bootstyle="danger-outline").pack(pady=5)

# --- Theme Toggle & Status Bar
theme_toggle_btn = ttk.Button(app, text="🌗 Toggle Theme", command=toggle_theme, bootstyle="dark")
theme_toggle_btn.pack(side="bottom", pady=5, fill="x")

status_label = ttk.Label(app, text="Ready.", font=("Helvetica", 10), anchor="w", bootstyle="info")
status_label.pack(side="bottom", fill="x")


app.mainloop()