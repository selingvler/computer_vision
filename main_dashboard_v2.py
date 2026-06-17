import tkinter as tk
from tkinter import messagebox, simpledialog
import subprocess

def run_registration():
    """Gets user data via GUI popup and runs register.py"""
    # Tkinter arayüzü üzerinden kullanıcıdan ID ve İsim alıyoruz
    user_id = simpledialog.askstring("Registration", "Enter User ID (Numbers only):", parent=root)
    if not user_id: # Kullanıcı iptale basarsa durdur
        return
        
    user_name = simpledialog.askstring("Registration", "Enter User Name:", parent=root)
    if not user_name:
        return

    try:
        # Aldığımız bu bilgileri register.py koduna gizlice (argüman olarak) gönderiyoruz
        subprocess.Popen(["python3", "register.py", user_id, user_name])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start Registration:\n{str(e)}")

def run_training():
    """Runs the train_model.py script."""
    try:
        messagebox.showinfo("Training Started", "System is learning faces. Please wait...")
        subprocess.run(["python3", "train_model.py"])
        messagebox.showinfo("Training Complete", "AI Model updated successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model:\n{str(e)}")

def run_security_system():
    """Runs the main security_system.py script."""
    try:
        subprocess.Popen(["python3", "security_system.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start Security System:\n{str(e)}")

# ==========================================
# GUI SETUP (Tkinter Window Design)
# ==========================================
root = tk.Tk()
root.title("Biometric AI Security - Admin Panel")
root.geometry("400x350")
root.configure(bg="#2c3e50")

# Header Label
header_label = tk.Label(root, text="ACCESS CONTROL\nSYSTEM", font=("Helvetica", 18, "bold"), fg="#ecf0f1", bg="#2c3e50", pady=20)
header_label.pack()

# Buttons
btn_register = tk.Button(root, text="1. Register New User", font=("Helvetica", 12), bg="#3498db", fg="black", width=25, height=2, command=run_registration)
btn_register.pack(pady=10)

btn_train = tk.Button(root, text="2. Train AI Model", font=("Helvetica", 12), bg="#f1c40f", fg="black", width=25, height=2, command=run_training)
btn_train.pack(pady=10)

btn_security = tk.Button(root, text="3. Start Live Security Feed", font=("Helvetica", 12, "bold"), bg="#e74c3c", fg="black", width=25, height=2, command=run_security_system)
btn_security.pack(pady=10)

# Footer
footer_label = tk.Label(root, text="CMP3011 - Final Project", font=("Helvetica", 9, "italic"), fg="#7f8c8d", bg="#2c3e50")
footer_label.pack(side=tk.BOTTOM, pady=10)

def on_closing():
    """Ensures all subprocesses are killed when the main window is closed."""
    import psutil
    parent = psutil.Process()
    for child in parent.children(recursive=True):
        child.kill()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()