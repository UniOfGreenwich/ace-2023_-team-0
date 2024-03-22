import tkinter as tk
from tkinter import messagebox, simpledialog, font
import hashlib
from BTC import Bitcoin  # Assuming BTC.py contains a class Bitcoin
from ETH import ETH      # Assuming ETH.py contains a class ETH



USER_DETAILS_FILEPATH = "users.txt"
PUNCTUATIONS = "@#$%&"


def hash_password(pwd):
    hashed_pwd = hashlib.sha256(pwd.encode('utf-8')).hexdigest()
    return hashed_pwd

def save_user(username, hashed_pwd):
    with open(USER_DETAILS_FILEPATH, "a") as f:
        f.write(f"{username} {hashed_pwd}\n")

def user_exists(username):
    try:
        with open(USER_DETAILS_FILEPATH, "r") as f:
            for line in f:
                parts = line.split()
                if parts[0] == username:
                    return True
    except FileNotFoundError:
        with open(USER_DETAILS_FILEPATH, "w") as f:
            pass
    return False

def authenticate_user(username, password):
    with open(USER_DETAILS_FILEPATH, "r") as f:
        for line in f:
            parts = line.split()
            if parts[0] == username and parts[1] == hash_password(password):
                return True
    return False

def register(username_entry, password_entry, retype_password_entry):
    username = username_entry.get()
    password = password_entry.get()
    retype_password = retype_password_entry.get()

    # Check for empty fields
    if not username or not password or not retype_password:
        messagebox.showerror("Error", "Username and password fields cannot be empty.")
        return

    if password != retype_password:
        messagebox.showerror("Error", "Passwords do not match.")
        return

    if user_exists(username):
        messagebox.showerror("Error", "User already exists.")
        return

    hashed_password = hash_password(password)
    save_user(username, hashed_password)
    messagebox.showinfo("Success", "User created successfully.\nRemember your password.")


def login(username_entry, password_entry, root):
    username = username_entry.get()
    password = password_entry.get()

    # Check for empty fields
    if not username or not password:
        messagebox.showerror("Error", "Username and password fields cannot be empty.")
        return

    if user_exists(username) and authenticate_user(username, password):
        messagebox.showinfo("Success", "Login successful.")
        UI(root)
    else:
        messagebox.showerror("Error", "Incorrect username or password.")

def create_login_ui(root):
    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Login")

    center_frame = tk.Frame(root, bg='#f7f7f7')
    center_frame.place(relx=0.5, rely=0.5, anchor='center')

    # Styling
    root.configure(bg='#f7f7f7')
    heading_font = font.Font(size=24, weight='bold')
    label_font = font.Font(size=12)
    entry_font = font.Font(size=12)

    # Heading
    heading_label = tk.Label(root, text="User Login", font=heading_font, bg='#f7f7f7')
    heading_label.pack(pady=20)

    form_frame = tk.Frame(root, bg='#f7f7f7')
    form_frame.pack(pady=10)

    # Username
    tk.Label(form_frame, text="Username:", font=label_font, bg='#f7f7f7').grid(row=0, column=0, sticky='e', padx=10, pady=10)
    username_entry = tk.Entry(form_frame, font=entry_font)
    username_entry.grid(row=0, column=1, padx=10, pady=10)

    # Password
    tk.Label(form_frame, text="Password:", font=label_font, bg='#f7f7f7').grid(row=1, column=0, sticky='e', padx=10, pady=10)
    password_entry = tk.Entry(form_frame, show="*", font=entry_font)
    password_entry.grid(row=1, column=1, padx=10, pady=10)

    # Login button
    login_button = style_button(tk.Button(center_frame, text="Login", command=lambda: login(username_entry, password_entry, root)))
    login_button.grid(row=2, column=1, pady=10)

    # Register button
    back_button = style_button(tk.Button(center_frame, text="Back to Register", command=lambda: create_register_ui(root)))
    back_button.grid(row=3, column=1, pady=10)

    #Exit button
    exit_button = style_button(tk.Button(center_frame, text="Exit", command=root.quit))
    exit_button.grid(row=4, column=1, pady=10)



def create_register_ui(root):
    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Register")
    
    center_frame = tk.Frame(root, bg='#f7f7f7')
    center_frame.place(relx=0.5, rely=0.5, anchor='center')

    # Styling
    root.configure(bg='#f7f7f7')
    heading_font = font.Font(size=24, weight='bold')
    label_font = font.Font(size=12)
    entry_font = font.Font(size=12)

    # Heading
    heading_label = tk.Label(root, text="Register New User", font=heading_font, bg='#f7f7f7')
    heading_label.pack(pady=20)

    form_frame = tk.Frame(root, bg='#f7f7f7')
    form_frame.pack(pady=10)

    # Username Entry
    tk.Label(form_frame, text="Username:", font=label_font, bg='#f7f7f7').grid(row=0, column=0, sticky='e', padx=10, pady=10)
    username_entry = tk.Entry(form_frame, font=entry_font)
    username_entry.grid(row=0, column=1, padx=10, pady=10)

    # Password Entry
    tk.Label(form_frame, text="Password:", font=label_font, bg='#f7f7f7').grid(row=1, column=0, sticky='e', padx=10, pady=10)
    password_entry = tk.Entry(form_frame, show="*", font=entry_font)
    password_entry.grid(row=1, column=1, padx=10, pady=10)

    # Retype Password Entry
    tk.Label(form_frame, text="Retype Password:", font=label_font, bg='#f7f7f7').grid(row=2, column=0, sticky='e', padx=10, pady=10)
    retype_password_entry = tk.Entry(form_frame, show="*", font=entry_font)
    retype_password_entry.grid(row=2, column=1, padx=10, pady=10)

    # Register actvity
    register_button = style_button(tk.Button(center_frame, text="Register", command=lambda: register(username_entry, password_entry, retype_password_entry)))
    register_button.grid(row=3, column=1, pady=10)

    #Login Button
    login_button = style_button(tk.Button(center_frame, text="Already have an account? Login", command=lambda: create_login_ui(root)))
    login_button.grid(row=4, column=1, pady=10)

    #Exit button
    exit_button = style_button(tk.Button(center_frame, text="Exit", command=root.quit))
    exit_button.grid(row=5, column=1, pady=10)

def UI(root):
    # Clear the window for the prediction UI
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Cryptocurrency Price Prediction Tool")

    # Center the content in the window
    content_frame = tk.Frame(root)
    content_frame.place(relx=0.5, rely=0.5, anchor='center')

    tk.Label(content_frame, text="Choose a cryptocurrency to predict its future price.", padx=10, pady=20).grid(row=0, column=0, columnspan=2)

    # Styling
    button_font = font.Font(size=14, weight='bold')
    button_style = {'font': button_font, 'bg': '#4CAF50', 'fg': 'white', 'padx': 20, 'pady': 10}

    btn_bitcoin = tk.Button(content_frame, text="Predict Bitcoin Price", command=lambda: Bitcoin().predict_price(), **button_style)
    btn_bitcoin.grid(row=1, column=0, padx=20, pady=20)

    btn_eth = tk.Button(content_frame, text="Predict ETH Price", command=lambda: ETH().predict_price(), **button_style)
    btn_eth.grid(row=1, column=1, padx=20, pady=20)

    # Exit Button
    exit_button = tk.Button(content_frame, text="Exit", command=root.quit, **button_style)
    exit_button.grid(row=2, column=0, columnspan=2)

    # Update window dimensions
    root.geometry('800x600')  # Wider window for better layout

def style_button(btn):
    btn.config(bg='#4CAF50', fg='white', padx=10, pady=5, font=('Helvetica', 12, 'bold'))
    return btn

def main():

    root = tk.Tk()
    root.state('zoomed')  # Maximize the main window
    create_register_ui(root)  # Create the register interface in the main window
    root.mainloop()

if __name__ == "__main__":
    main()