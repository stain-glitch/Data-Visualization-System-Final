import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sqlite3
from datetime import datetime
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import json
import time
import gc
import dask.dataframe as dd
import threading

plt.style.use('seaborn-v0_8-darkgrid')

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users 
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE, 
              password TEXT,
              role TEXT,
              created_at TIMESTAMP)''')

c.execute('''CREATE TABLE IF NOT EXISTS visualizations
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              chart_type TEXT,
              created_at TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS user_preferences
             (user_id INTEGER PRIMARY KEY,
              last_file_path TEXT,
              x_column TEXT,
              y_column TEXT,
              chart_type TEXT,
              chart_title TEXT,
              show_grid INTEGER,
              show_legend INTEGER,
              FOREIGN KEY(user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS saved_work
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              name TEXT,
              file_path TEXT,
              x_column TEXT,
              y_column TEXT,
              chart_type TEXT,
              chart_title TEXT,
              show_grid INTEGER,
              show_legend INTEGER,
              created_at TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS dashboards
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              name TEXT,
              layout TEXT,  -- JSON array of chart configurations
              created_at TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id))''')
conn.commit()
conn.close()

class ModernCard(ttk.Frame):
    """Modern card component with shadow effect and rounded appearance"""
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.configure(relief='flat', borderwidth=0)
        
        if title:
            header_frame = ttk.Frame(self)
            header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
            
            title_label = ttk.Label(header_frame, text=title, font=('Segoe UI', 12, 'bold'))
            title_label.pack(anchor='w')
            
            separator = ttk.Separator(header_frame, orient='horizontal')
            separator.pack(fill=tk.X, pady=(10, 0))
        
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

class LoadingScreen:
    """Shows loading screen during long operations"""
    def __init__(self, parent, message="Loading..."):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Processing")
        self.window.geometry("400x150")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.resizable(False, False)
        
        # Center the window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'+{x}+{y}')
        
        # Content
        container = ttk.Frame(self.window, padding=20)
        container.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(container, text=message, font=('Segoe UI', 12)).pack(pady=(0, 20))
        
        self.progress = ttk.Progressbar(container, mode='indeterminate', length=300)
        self.progress.pack()
        self.progress.start(10)
        
        self.status_label = ttk.Label(container, text="", font=('Segoe UI', 9))
        self.status_label.pack(pady=(10, 0))
        
    def update_status(self, message):
        """Update the status message"""
        self.status_label.config(text=message)
        self.window.update()
        
    def close(self):
        """Close the loading screen"""
        self.window.destroy()

class DashboardChartCard(ttk.Frame):
    """Card for each chart in a dashboard"""
    def __init__(self, parent, chart_config, dashboard, **kwargs):
        super().__init__(parent, **kwargs)
        self.chart_config = chart_config
        self.dashboard = dashboard
        
        self.configure(relief='raised', borderwidth=1, padding=5)
        
        # Header with title and actions
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(header_frame, text=chart_config['title'], 
                               font=('Segoe UI', 10, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        edit_btn = ttk.Button(btn_frame, text="✏️", width=3, 
                             command=lambda: self.dashboard.edit_chart(self))
        edit_btn.pack(side=tk.LEFT, padx=2)
        
        remove_btn = ttk.Button(btn_frame, text="❌", width=3, 
                               command=lambda: self.dashboard.remove_chart(self))
        remove_btn.pack(side=tk.LEFT, padx=2)
        
        # Chart area
        self.figure = plt.Figure(figsize=(4, 3), dpi=80)
        self.figure.patch.set_facecolor('#ffffff')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Generate the chart
        self.generate_chart()
    
    def generate_chart(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get data from dashboard
        data = self.dashboard.data
        
        if data is None:
            ax.text(0.5, 0.5, 'No data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=10, color='#6c757d')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.canvas.draw()
            return
        
        try:
            x_col = self.chart_config['x_column']
            y_col = self.chart_config['y_column']
            chart_type = self.chart_config['chart_type']
            title = self.chart_config['title']
            show_grid = self.chart_config['show_grid']
            show_legend = self.chart_config['show_legend']
            
            colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14', '#20c997', '#6c757d']
            
            # For large datasets, use optimized plotting
            if chart_type == "bar":
                # For large datasets, sample or aggregate
                if len(data) > 10000:
                    sampled = data.sample(min(1000, len(data)))
                    sampled.plot(kind='bar', x=x_col, y=y_col, ax=ax, 
                                legend=show_legend, color=colors[0])
                else:
                    data.plot(kind='bar', x=x_col, y=y_col, ax=ax, 
                            legend=show_legend, color=colors[0])
            elif chart_type == "line":
                # For large time series, downsample
                if len(data) > 10000 and is_datetime64_any_dtype(data[x_col]):
                    data = data.set_index(x_col).resample('D').mean().reset_index()
                data.plot(kind='line', x=x_col, y=y_col, ax=ax, 
                        legend=show_legend, color=colors[0], linewidth=2)
            elif chart_type == "scatter":
                # Sample for large datasets
                if len(data) > 5000:
                    sampled = data.sample(min(2000, len(data)))
                    sampled.plot(kind='scatter', x=x_col, y=y_col, ax=ax, 
                                color=colors[0], alpha=0.7)
                else:
                    data.plot(kind='scatter', x=x_col, y=y_col, ax=ax, 
                            color=colors[0], alpha=0.7)
            elif chart_type == "histogram":
                # For large datasets, use more bins
                bins = 50 if len(data) > 10000 else 20
                data[y_col].plot(kind='hist', ax=ax, legend=show_legend, 
                                color=colors[0], alpha=0.8, bins=bins)
                ax.set_xlabel(y_col)
            elif chart_type == "box":
                # For large datasets, use sampled version
                if len(data) > 10000:
                    sampled = data.sample(min(5000, len(data)))
                    sampled[y_col].plot(kind='box', ax=ax, color=colors[0])
                else:
                    data[y_col].plot(kind='box', ax=ax, color=colors[0])
            elif chart_type == "pie":
                # Only show top categories for large datasets
                top_categories = 10
                if len(data) > 1000:
                    top_categories = 8
                pie_data = data.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(top_categories)
                pie_data.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)

            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
            if chart_type not in ["pie", "box"]:
                ax.set_xlabel(x_col, fontsize=8)
                ax.set_ylabel(y_col, fontsize=8)

            ax.grid(show_grid, alpha=0.3)
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=10, color='red')
            self.canvas.draw()

class LoginWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualization Platform")
        self.root.geometry("450x550")
        self.root.configure(bg="#f8f9fa")
        
        # Set close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.center_window()
        
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 30))
        
        title_label = ttk.Label(title_frame, text="DataViz System", 
                               font=('Segoe UI', 24, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Sign in to your account", 
                                  font=('Segoe UI', 11), foreground='#6c757d')
        subtitle_label.pack(pady=(5, 0))
        
        # Login card
        login_card = ModernCard(main_container)
        login_card.pack(fill=tk.X, pady=(0, 20))
        
        # Username field
        ttk.Label(login_card.content_frame, text="Username", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.username = ttk.Entry(login_card.content_frame, font=('Segoe UI', 11))
        self.username.pack(fill=tk.X, pady=(0, 15), ipady=8)
        
        # Password field
        ttk.Label(login_card.content_frame, text="Password", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.password = ttk.Entry(login_card.content_frame, show="*", font=('Segoe UI', 11))
        self.password.pack(fill=tk.X, pady=(0, 20), ipady=8)
        
        # Login button
        login_btn = ttk.Button(login_card.content_frame, text="Sign In", command=self.login)
        login_btn.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        # Register button
        register_btn = ttk.Button(login_card.content_frame, text="Create Account", command=self.register)
        register_btn.pack(fill=tk.X, ipady=8)
        
        # Focus on username
        self.username.focus()
        
        # Bind Enter key
        self.root.bind('<Return>', lambda e: self.login())
    
    def on_close(self):
        """Handle window close event"""
        self.root.destroy()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def login(self):
        username = self.username.get()
        password = self.password.get()

        if not username or not password:
            messagebox.showerror("Error", "Please fill in all fields")
            return

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, role FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            # Destroy login widgets and create dashboard
            for widget in self.root.winfo_children():
                widget.destroy()
            app = DataVisualizationDashboard(self.root, user_id=user[0], user_role=user[1])
        else:
            messagebox.showerror("Error", "Invalid credentials")

    def register(self):
        RegisterWindow()

class RegisterWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Create Account")
        self.window.geometry("450x600")
        self.window.configure(bg="#f8f9fa")
        
        # Center window
        self.center_window()
        
        # Main container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 30))
        
        title_label = ttk.Label(title_frame, text="Create Account", 
                               font=('Segoe UI', 24, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Join our data visualization platform", 
                                  font=('Segoe UI', 11), foreground='#6c757d')
        subtitle_label.pack(pady=(5, 0))
        
        # Register card
        register_card = ModernCard(main_container)
        register_card.pack(fill=tk.X, pady=(0, 20))
        
        # Username field
        ttk.Label(register_card.content_frame, text="Username", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.username = ttk.Entry(register_card.content_frame, font=('Segoe UI', 11))
        self.username.pack(fill=tk.X, pady=(0, 15), ipady=8)
        
        # Password field
        ttk.Label(register_card.content_frame, text="Password", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.password = ttk.Entry(register_card.content_frame, show="*", font=('Segoe UI', 11))
        self.password.pack(fill=tk.X, pady=(0, 15), ipady=8)
        
        # Admin code field
        ttk.Label(register_card.content_frame, text="Admin Code (Optional)", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.admin_code = ttk.Entry(register_card.content_frame, font=('Segoe UI', 11))
        self.admin_code.pack(fill=tk.X, pady=(0, 15), ipady=8)
        
        # Help text
        help_label = ttk.Label(register_card.content_frame, 
                              text="Leave admin code blank for regular user account", 
                              font=('Segoe UI', 9), foreground='#6c757d')
        help_label.pack(pady=(0, 20))
        
        # Register button
        register_btn = ttk.Button(register_card.content_frame, text="Create Account", 
                                 command=self.register_user)
        register_btn.pack(fill=tk.X, ipady=8)
        
        # Focus on username
        self.username.focus()

    def center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def register_user(self):
        username = self.username.get()
        password = self.password.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please fill in username and password")
            return
            
        role = "admin" if self.admin_code.get() == "admin123" else "user"

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)",
                     (username, password, role, datetime.now()))
            conn.commit()
            messagebox.showinfo("Success", f"Account created successfully as {role}")
            self.window.destroy()
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists")
        finally:
            conn.close()

class DataVisualizationDashboard:
    def __init__(self, root, user_id, user_role):
        self.root = root
        self.user_id = user_id
        self.user_role = user_role
        self.root.title("Data Visualization Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f8f9fa")
        self.root.state('zoomed')  # Maximize window on Windows
        
        # Set close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data = None
        self.columns = []
        self.numeric_columns = []
        self.current_file_path = None
        self.current_dashboard = None
        self.dashboard_charts = []  # List of chart cards in dashboard
        self.large_dataset = False  # Flag for large datasets
        self.sampling_size = 10000  # Default sampling size
        self.dask_data = None  # For handling very large datasets

        self.setup_styles()
        
        self.create_header()
        
        self.create_main_content()
        
        # Load user preferences
        self.load_preferences()

    def setup_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.configure('Card.TFrame', background='white', relief='flat')
        style.configure('Primary.TButton', font=('Segoe UI', 10))
        style.configure('Secondary.TButton', font=('Segoe UI', 10))
        style.configure('Dashboard.TFrame', background='#f0f2f5')
        style.configure('ChartCard.TFrame', background='white', relief='raised', borderwidth=1)

    def create_header(self):
        """Create modern header with user info and navigation"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        
        # Left side - Title
        left_frame = ttk.Frame(header_frame)
        left_frame.pack(side=tk.LEFT)
        
        title_label = ttk.Label(left_frame, text="Main Dashboard", 
                               font=('Segoe UI', 20, 'bold'))
        title_label.pack(anchor='w')
        
        subtitle_label = ttk.Label(left_frame, text="Create stunning visualizations from your data", 
                                  font=('Segoe UI', 11), foreground='#6c757d')
        subtitle_label.pack(anchor='w')
        
        # Right side - User info
        right_frame = ttk.Frame(header_frame)
        right_frame.pack(side=tk.RIGHT)
        
        user_info_card = ModernCard(right_frame)
        user_info_card.pack()
        
        role_label = ttk.Label(user_info_card.content_frame, 
                              text=f"Role: {self.user_role.title()}", 
                              font=('Segoe UI', 10, 'bold'))
        role_label.pack()
        
        # Memory status
        self.memory_label = ttk.Label(user_info_card.content_frame, 
                                     text="Memory: 0 MB", 
                                     font=('Segoe UI', 9), foreground='#6c757d')
        self.memory_label.pack(pady=(5, 0))
        
        logout_btn = ttk.Button(user_info_card.content_frame, text="🚪 Logout", 
                               command=self.logout, style='Secondary.TButton')
        logout_btn.pack(pady=(10, 0), ipady=4)

    def create_main_content(self):
        """Create main content area with modern card layout"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left sidebar with controls
        sidebar_frame = ttk.Frame(main_container)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Scrollable sidebar
        self.create_sidebar(sidebar_frame)
        
        # Right side - Tabbed content area
        self.tab_control = ttk.Notebook(main_container)
        self.tab_control.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Single Chart Tab
        self.single_chart_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.single_chart_tab, text="📊 Single Chart")
        self.create_single_chart_tab(self.single_chart_tab)
        
        # Dashboard Tab
        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.dashboard_tab, text="📈 Dashboard")
        self.create_dashboard_tab(self.dashboard_tab)

    def create_single_chart_tab(self, parent):
        """Create single chart visualization area"""
        viz_card = ModernCard(parent, title="📈 Visualization")
        viz_card.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.figure.patch.set_facecolor('#ffffff')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_card.content_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, '📊 Load data to begin visualization', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=self.ax.transAxes, fontsize=16, color='#6c757d')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    def create_dashboard_tab(self, parent):
        """Create dashboard area with multiple charts"""
        # Dashboard management controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Dashboard selection
        ttk.Label(control_frame, text="Dashboard:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.dashboard_var = tk.StringVar()
        self.dashboard_combobox = ttk.Combobox(control_frame, textvariable=self.dashboard_var, 
                                              width=25, state="readonly")
        self.dashboard_combobox.pack(side=tk.LEFT, padx=5)
        self.dashboard_combobox.bind('<<ComboboxSelected>>', self.load_dashboard)
        
        # Dashboard actions
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        new_dashboard_btn = ttk.Button(btn_frame, text="🆕 New Dashboard", 
                                      command=self.create_new_dashboard, width=18)
        new_dashboard_btn.pack(side=tk.LEFT, padx=2)
        
        rename_dashboard_btn = ttk.Button(btn_frame, text="✏️ Rename", 
                                        command=self.rename_dashboard, width=10)
        rename_dashboard_btn.pack(side=tk.LEFT, padx=2)
        
        delete_dashboard_btn = ttk.Button(btn_frame, text="🗑️ Delete", 
                                        command=self.delete_dashboard, width=10)
        delete_dashboard_btn.pack(side=tk.LEFT, padx=2)
        
        # Dashboard canvas with scrollbar
        self.dashboard_container = ttk.Frame(parent, style='Dashboard.TFrame')
        self.dashboard_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for scrolling
        self.dashboard_canvas = tk.Canvas(self.dashboard_container, bg='#f0f2f5')
        self.dashboard_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.dashboard_container, orient="vertical", 
                                 command=self.dashboard_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas scrolling
        self.dashboard_canvas.configure(yscrollcommand=scrollbar.set)
        self.dashboard_canvas.bind('<Configure>', lambda e: self.dashboard_canvas.configure(scrollregion=self.dashboard_canvas.bbox("all")))
        
        # Frame for dashboard content
        self.dashboard_content = ttk.Frame(self.dashboard_canvas, style='Dashboard.TFrame')
        self.dashboard_canvas.create_window((0, 0), window=self.dashboard_content, anchor="nw")
        
        # Mouse wheel scrolling
        self.dashboard_canvas.bind_all("<MouseWheel>", lambda e: self.dashboard_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Add chart button
        add_chart_frame = ttk.Frame(parent)
        add_chart_frame.pack(fill=tk.X, padx=10, pady=10)
        
        add_chart_btn = ttk.Button(add_chart_frame, text="➕ Add Chart to Dashboard", 
                                  command=self.add_chart_to_dashboard, style='Primary.TButton')
        add_chart_btn.pack(fill=tk.X, ipady=8)
        
        # Load dashboards
        self.load_dashboards_list()

    def create_sidebar(self, parent):
        """Create modern sidebar with control cards"""
        # Scrollable container
        canvas = tk.Canvas(parent, width=350, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", 
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Data Source Card
        data_card = ModernCard(scrollable_frame, title="📁 Data Source")
        data_card.pack(fill=tk.X, pady=(0, 20))
        
        load_btn = ttk.Button(data_card.content_frame, text="📂 Load Excel/CSV File", 
                             command=self.load_file, style='Primary.TButton')
        load_btn.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        # NEW: Large dataset options
        large_options_frame = ttk.Frame(data_card.content_frame)
        large_options_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(large_options_frame, text="Large Dataset Options:", 
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        
        options_frame = ttk.Frame(large_options_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Sampling size
        ttk.Label(options_frame, text="Sampling Size:", 
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.sampling_var = tk.StringVar(value="10000")
        sampling_entry = ttk.Entry(options_frame, textvariable=self.sampling_var, 
                                  width=8, font=('Segoe UI', 9))
        sampling_entry.pack(side=tk.LEFT)
        
        # Clear data button
        clear_btn = ttk.Button(large_options_frame, text="🗑️ Clear Data", 
                              command=self.clear_data, style='Secondary.TButton')
        clear_btn.pack(fill=tk.X, pady=(5, 0), ipady=4)
        
        self.file_status_frame = ttk.Frame(data_card.content_frame)
        self.file_status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.file_label = ttk.Label(self.file_status_frame, text="No file loaded", 
                                   foreground='#6c757d', font=('Segoe UI', 10))
        self.file_label.pack()

        # Chart Type Card
        chart_card = ModernCard(scrollable_frame, title="📊 Chart Type")
        chart_card.pack(fill=tk.X, pady=(0, 20))
        
        self.chart_type = tk.StringVar(value="bar")
        chart_types = [
            ("📊 Bar Chart", "bar"), 
            ("📈 Line Chart", "line"),
            ("🎯 Scatter Plot", "scatter"), 
            ("📉 Histogram", "histogram"),
            ("📦 Box Plot", "box"), 
            ("🥧 Pie Chart", "pie")
        ]
        
        for text, value in chart_types:
            rb = ttk.Radiobutton(chart_card.content_frame, text=text, value=value, 
                                variable=self.chart_type)
            rb.pack(anchor=tk.W, pady=2)
        
        self.recommend_label = ttk.Label(chart_card.content_frame, 
                                        text="💡 Load data for smart recommendations", 
                                        foreground="#007bff", font=('Segoe UI', 9))
        self.recommend_label.pack(pady=(10, 0))

        # Data Selection Card
        selection_card = ModernCard(scrollable_frame, title="🎯 Data Selection")
        selection_card.pack(fill=tk.X, pady=(0, 20))
        
        # X-Axis
        ttk.Label(selection_card.content_frame, text="X-Axis Column", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.x_column = tk.StringVar()
        self.x_combobox = ttk.Combobox(selection_card.content_frame, textvariable=self.x_column, 
                                      state="readonly", font=('Segoe UI', 10))
        self.x_combobox.pack(fill=tk.X, pady=(0, 15), ipady=5)
        
        # Y-Axis
        ttk.Label(selection_card.content_frame, text="Y-Axis Column", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.y_column = tk.StringVar()
        self.y_combobox = ttk.Combobox(selection_card.content_frame, textvariable=self.y_column, 
                                      state="readonly", font=('Segoe UI', 10))
        self.y_combobox.pack(fill=tk.X, pady=(0, 10), ipady=5)

        # Chart Options Card
        options_card = ModernCard(scrollable_frame, title="⚙️ Chart Options")
        options_card.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        ttk.Label(options_card.content_frame, text="Chart Title", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.chart_title = tk.StringVar(value="Data Visualization")
        title_entry = ttk.Entry(options_card.content_frame, textvariable=self.chart_title, 
                               font=('Segoe UI', 10))
        title_entry.pack(fill=tk.X, pady=(0, 15), ipady=5)
        
        # Options
        self.grid_lines = tk.BooleanVar(value=True)
        cb1 = ttk.Checkbutton(options_card.content_frame, text="📏 Show Grid Lines", 
                             variable=self.grid_lines)
        cb1.pack(anchor=tk.W, pady=2)
        
        self.legend = tk.BooleanVar(value=True)
        cb2 = ttk.Checkbutton(options_card.content_frame, text="🏷️ Show Legend", 
                             variable=self.legend)
        cb2.pack(anchor=tk.W, pady=2)

        # Actions Card
        actions_card = ModernCard(scrollable_frame, title="🚀 Actions")
        actions_card.pack(fill=tk.X, pady=(0, 20))
        
        generate_btn = ttk.Button(actions_card.content_frame, text="✨ Generate Chart", 
                                 command=self.generate_chart, style='Primary.TButton')
        generate_btn.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        export_btn = ttk.Button(actions_card.content_frame, text="📤 Export Chart", 
                             command=self.export_chart, style='Secondary.TButton')
        export_btn.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        save_btn = ttk.Button(actions_card.content_frame, text="💾 Save Work", 
                             command=self.save_work, style='Primary.TButton')
        save_btn.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        load_work_btn = ttk.Button(actions_card.content_frame, text="📂 Load Saved Work", 
                                  command=self.load_saved_work, style='Secondary.TButton')
        load_work_btn.pack(fill=tk.X, ipady=8)

        # Admin Tools Card
        if self.user_role == "admin":
            admin_card = ModernCard(scrollable_frame, title="👑 Admin Tools")
            admin_card.pack(fill=tk.X, pady=(0, 20))
            
            admin_btn = ttk.Button(admin_card.content_frame, text="📊 Generate Report", 
                                  command=self.generate_admin_report)
            admin_btn.pack(fill=tk.X, ipady=8)

        # Setup recommendation triggers
        self.x_column.trace_add('write', lambda *_: self.recommend_chart_type())
        self.y_column.trace_add('write', lambda *_: self.recommend_chart_type())

    # DASHBOARD MANAGEMENT FUNCTIONS
    
    def load_dashboards_list(self):
        """Load available dashboards for the user"""
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT id, name FROM dashboards WHERE user_id=?", (self.user_id,))
            dashboards = c.fetchall()
            conn.close()
            
            # Update combobox
            self.dashboard_combobox['values'] = [name for _, name in dashboards]
            self.dashboard_ids = {name: id for id, name in dashboards}
            
            # Load first dashboard if available
            if dashboards:
                self.dashboard_var.set(dashboards[0][1])
                self.load_dashboard()
        except Exception as e:
            print("Error loading dashboards:", e)

    def create_new_dashboard(self):
        """Create a new dashboard"""
        name = simpledialog.askstring("New Dashboard", "Enter dashboard name:")
        if not name:
            return
            
        # Create new dashboard in database
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO dashboards (user_id, name, layout, created_at) VALUES (?, ?, ?, ?)",
                     (self.user_id, name, '[]', datetime.now()))
            conn.commit()
            conn.close()
            
            # Refresh dashboard list
            self.load_dashboards_list()
            self.dashboard_var.set(name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dashboard: {str(e)}")

    def rename_dashboard(self):
        """Rename the current dashboard"""
        if not self.current_dashboard:
            messagebox.showwarning("Warning", "No dashboard selected")
            return
            
        new_name = simpledialog.askstring("Rename Dashboard", "Enter new name:", 
                                         initialvalue=self.dashboard_var.get())
        if not new_name:
            return
            
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("UPDATE dashboards SET name=? WHERE id=?", 
                     (new_name, self.current_dashboard['id']))
            conn.commit()
            conn.close()
            
            # Refresh dashboard list
            self.load_dashboards_list()
            self.dashboard_var.set(new_name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rename dashboard: {str(e)}")

    def delete_dashboard(self):
        """Delete the current dashboard"""
        if not self.current_dashboard:
            messagebox.showwarning("Warning", "No dashboard selected")
            return
            
        if not messagebox.askyesno("Confirm Delete", 
                                  f"Are you sure you want to delete '{self.dashboard_var.get()}'?"):
            return
            
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("DELETE FROM dashboards WHERE id=?", (self.current_dashboard['id'],))
            conn.commit()
            conn.close()
            
            # Clear current dashboard and refresh list
            self.current_dashboard = None
            self.clear_dashboard()
            self.load_dashboards_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete dashboard: {str(e)}")

    def load_dashboard(self, event=None):
        """Load the selected dashboard"""
        dashboard_name = self.dashboard_var.get()
        if not dashboard_name:
            return
            
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT id, layout FROM dashboards WHERE user_id=? AND name=?", 
                     (self.user_id, dashboard_name))
            dashboard = c.fetchone()
            conn.close()
            
            if dashboard:
                self.current_dashboard = {
                    'id': dashboard[0],
                    'name': dashboard_name,
                    'layout': json.loads(dashboard[1])
                }
                self.render_dashboard()
            else:
                messagebox.showwarning("Not Found", f"Dashboard '{dashboard_name}' not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dashboard: {str(e)}")

    def save_dashboard(self):
        """Save the current dashboard layout"""
        if not self.current_dashboard:
            messagebox.showwarning("Warning", "No dashboard selected")
            return
            
        # Get current layout configuration
        layout = []
        for chart_card in self.dashboard_charts:
            layout.append(chart_card.chart_config)
            
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("UPDATE dashboards SET layout=? WHERE id=?", 
                     (json.dumps(layout), self.current_dashboard['id']))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Dashboard saved successfully! 💾")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dashboard: {str(e)}")

    def render_dashboard(self):
        """Render the dashboard with all charts"""
        if not self.current_dashboard or not self.current_dashboard['layout']:
            self.clear_dashboard()
            return
            
        self.clear_dashboard()
        
        # Create grid layout
        grid_frame = ttk.Frame(self.dashboard_content)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Determine grid size (2 columns)
        num_charts = len(self.current_dashboard['layout'])
        num_rows = (num_charts + 1) // 2
        
        # Create charts in grid
        for i, chart_config in enumerate(self.current_dashboard['layout']):
            row = i // 2
            col = i % 2
            
            # Create chart card
            chart_frame = ttk.Frame(grid_frame, style='ChartCard.TFrame')
            chart_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Configure grid weights
            grid_frame.rowconfigure(row, weight=1)
            grid_frame.columnconfigure(col, weight=1)
            
            # Create chart
            chart_card = DashboardChartCard(chart_frame, chart_config, self)
            chart_card.pack(fill=tk.BOTH, expand=True)
            self.dashboard_charts.append(chart_card)
        
        # Save button
        save_btn = ttk.Button(self.dashboard_content, text="💾 Save Dashboard", 
                             command=self.save_dashboard, style='Primary.TButton')
        save_btn.pack(pady=10, ipady=8)

    def clear_dashboard(self):
        """Clear all charts from the dashboard"""
        for widget in self.dashboard_content.winfo_children():
            widget.destroy()
        self.dashboard_charts = []

    def add_chart_to_dashboard(self):
        """Add current chart configuration to the dashboard"""
        if not self.current_dashboard:
            messagebox.showwarning("Warning", "Please select or create a dashboard first")
            return
            
        if not self.data:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        # Create chart configuration
        chart_config = {
            'x_column': self.x_column.get(),
            'y_column': self.y_column.get(),
            'chart_type': self.chart_type.get(),
            'title': self.chart_title.get(),
            'show_grid': self.grid_lines.get(),
            'show_legend': self.legend.get()
        }
        
        # Add to dashboard layout
        self.current_dashboard['layout'].append(chart_config)
        
        # Re-render dashboard
        self.render_dashboard()
        
        # Auto-save dashboard
        self.save_dashboard()

    def edit_chart(self, chart_card):
        """Edit an existing chart in the dashboard"""
        if not self.current_dashboard:
            return
            
        # Update UI to match chart configuration
        config = chart_card.chart_config
        self.x_column.set(config['x_column'])
        self.y_column.set(config['y_column'])
        self.chart_type.set(config['chart_type'])
        self.chart_title.set(config['title'])
        self.grid_lines.set(config['show_grid'])
        self.legend.set(config['show_legend'])
        
        # Switch to single chart tab
        self.tab_control.select(0)
        
        # Generate the chart for editing
        self.generate_chart()
        
        # Show message
        messagebox.showinfo("Edit Chart", 
                          "Chart configuration loaded. Make changes and click 'Add to Dashboard' when ready. "
                          "Don't forget to save the dashboard after updating.")

    def remove_chart(self, chart_card):
        """Remove a chart from the dashboard"""
        if not self.current_dashboard:
            return
            
        # Remove from layout
        self.current_dashboard['layout'] = [config for config in self.current_dashboard['layout'] 
                                           if config != chart_card.chart_config]
        
        # Re-render dashboard
        self.render_dashboard()
        
        # Auto-save dashboard
        self.save_dashboard()

    # DATA HANDLING AND VISUALIZATION FUNCTIONS

    def load_preferences(self):
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM user_preferences WHERE user_id=?", (self.user_id,))
            prefs = c.fetchone()
            conn.close()
            
            if prefs:
                # Unpack preferences
                last_file_path = prefs[1]
                x_col = prefs[2]
                y_col = prefs[3]
                chart_type = prefs[4]
                chart_title = prefs[5]
                show_grid = bool(prefs[6])
                show_legend = bool(prefs[7])
                
                # Set UI values
                self.chart_type.set(chart_type)
                self.chart_title.set(chart_title)
                self.grid_lines.set(show_grid)
                self.legend.set(show_legend)
                
                # Try to load last file
                if last_file_path and os.path.exists(last_file_path):
                    self.load_file_from_path(last_file_path)
                    
                    # Set column selections if they exist
                    if x_col in self.columns:
                        self.x_column.set(x_col)
                    if y_col and y_col in self.numeric_columns:
                        self.y_column.set(y_col)
                        
                    self.generate_chart()
                elif last_file_path:
                    messagebox.showinfo("File Not Found", 
                                       f"The last used file was not found:\n{last_file_path}")
        except Exception as e:
            print("Error loading preferences:", e)

    def save_preferences(self):
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            # Delete existing preferences
            c.execute("DELETE FROM user_preferences WHERE user_id=?", (self.user_id,))
            
            # Insert new preferences if we have a file loaded
            if self.current_file_path:
                c.execute('''INSERT INTO user_preferences 
                            (user_id, last_file_path, x_column, y_column, chart_type, 
                             chart_title, show_grid, show_legend)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (self.user_id, 
                          self.current_file_path,
                          self.x_column.get(),
                          self.y_column.get(),
                          self.chart_type.get(),
                          self.chart_title.get(),
                          int(self.grid_lines.get()),
                          int(self.legend.get())))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print("Error saving preferences:", e)

    def logout(self):
        # Save preferences before logging out
        self.save_preferences()
        
        # Clear current dashboard
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Return to login screen
        LoginWindow(self.root)

    def on_close(self):
        self.save_preferences()
        self.root.destroy()

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Excel Files", "*.xlsx *.xls"), ("CSV Files", "*.csv"), 
                       ("Parquet Files", "*.parquet"), ("All Files", "*.*")]
        )
        if not file_path:
            return
            
        # Start loading in a separate thread
        threading.Thread(target=self.load_file_async, args=(file_path,)).start()

    def load_file_async(self, file_path):
        """Load file asynchronously with progress feedback"""
        loading_screen = LoadingScreen(self.root, "Loading data file...")
        
        try:
            # Get sampling size from UI
            try:
                self.sampling_size = int(self.sampling_var.get())
            except ValueError:
                self.sampling_size = 10000
                self.sampling_var.set("10000")
                
            loading_screen.update_status(f"Loading {os.path.basename(file_path)}...")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Handle large datasets
            if file_extension == '.parquet':
                # Use Dask for parallel processing of large Parquet files
                loading_screen.update_status("Using Dask for parallel processing...")
                self.dask_data = dd.read_parquet(file_path)
                
                # Compute to pandas if small enough
                if len(self.dask_data) < 500000:
                    loading_screen.update_status("Converting to pandas DataFrame...")
                    self.data = self.dask_data.compute()
                    self.dask_data = None
                else:
                    self.data = None
            else:
                # For CSV/Excel, use optimized pandas reading
                if file_extension in ['.xlsx', '.xls']:
                    loading_screen.update_status("Reading Excel file...")
                    self.data = pd.read_excel(file_path)
                elif file_extension == '.csv':
                    loading_screen.update_status("Reading CSV file...")
                    try:
                        # Try to read with chunks if large
                        if os.path.getsize(file_path) > 100000000:  # >100MB
                            loading_screen.update_status("Large file detected - using chunked reading...")
                            chunks = []
                            for chunk in pd.read_csv(file_path, chunksize=10000, low_memory=False, encoding='utf-8'):
                                chunks.append(chunk)
                                if len(chunks) > 50:  # Limit to 500,000 rows
                                    break
                            self.data = pd.concat(chunks, ignore_index=True)
                        else:
                            try:
                                self.data = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                            except UnicodeDecodeError:
                                self.data = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
                    except Exception as e:
                        loading_screen.update_status(f"Error: {str(e)}")
                        time.sleep(2)
                        raise e
            
            # Process the data
            self.process_loaded_data(file_path, loading_screen)
            
            # Generate chart
            self.generate_chart()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
        finally:
            loading_screen.close()
            self.update_memory_usage()

    def process_loaded_data(self, file_path, loading_screen):
        """Process the loaded data"""
        if self.data is not None:
            # Handle large datasets
            loading_screen.update_status("Processing data...")
            
            # Drop empty columns
            self.data.dropna(axis=1, how='all', inplace=True)
            
            # Sample large datasets
            if len(self.data) > self.sampling_size:
                loading_screen.update_status(f"Sampling {self.sampling_size} rows...")
                self.data = self.data.sample(self.sampling_size)
                self.large_dataset = True
            else:
                self.large_dataset = False
                
            file_name = os.path.basename(file_path)
            
            # Update file status
            self.root.after(0, self.update_file_status, file_name)
            
            # Extract columns
            self.columns = list(self.data.columns)
            self.numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
            
            # Update UI
            self.root.after(0, self.update_column_comboboxes)
            
            # Store current file path
            self.current_file_path = file_path
            self.recommend_chart_type()
            
        elif self.dask_data is not None:
            # Handle Dask dataframe
            loading_screen.update_status("Processing large dataset...")
            file_name = os.path.basename(file_path)
            
            # Update file status
            self.root.after(0, self.update_file_status, file_name, large=True)
            
            # Extract columns
            self.columns = self.dask_data.columns.tolist()
            numeric_cols = [col for col in self.columns if 
                           np.issubdtype(self.dask_data[col].dtype, np.number)]
            self.numeric_columns = numeric_cols
            
            # Update UI
            self.root.after(0, self.update_column_comboboxes)
            
            # Store current file path
            self.current_file_path = file_path
            self.large_dataset = True

    def update_file_status(self, file_name, large=False):
        """Update file status in UI"""
        for widget in self.file_status_frame.winfo_children():
            widget.destroy()
                
        status_frame = ttk.Frame(self.file_status_frame)
        status_frame.pack(fill=tk.X)
        
        # Success indicator
        success_label = ttk.Label(status_frame, text="✅", font=('Segoe UI', 12))
        success_label.pack(side=tk.LEFT)
        
        # File info
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        ttk.Label(info_frame, text=f"File: {file_name}", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        
        if self.data is not None:
            size_info = f"📊 {len(self.data)} rows, {len(self.data.columns)} columns"
        else:
            size_info = f"📊 Large dataset ({len(self.dask_data):,} rows)"
            
        if self.large_dataset:
            size_info += " (sampled)"
            
        ttk.Label(info_frame, text=size_info, 
                 font=('Segoe UI', 9), foreground='#6c757d').pack(anchor='w')

    def update_column_comboboxes(self):
        """Update column comboboxes in UI"""
        self.x_combobox['values'] = self.columns
        self.y_combobox['values'] = self.numeric_columns

        if self.columns:
            self.x_column.set(self.columns[0])
        if self.numeric_columns:
            self.y_column.set(self.numeric_columns[0])

    def clear_data(self):
        """Clear current data to free memory"""
        self.data = None
        self.dask_data = None
        self.columns = []
        self.numeric_columns = []
        self.current_file_path = None
        self.large_dataset = False
        
        # Update UI
        for widget in self.file_status_frame.winfo_children():
            widget.destroy()
            
        self.file_label = ttk.Label(self.file_status_frame, text="No file loaded", 
                                   foreground='#6c757d', font=('Segoe UI', 10))
        self.file_label.pack()
        
        self.x_combobox['values'] = []
        self.y_combobox['values'] = []
        self.x_column.set("")
        self.y_column.set("")
        
        # Reset chart
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, '📊 Load data to begin visualization', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=self.ax.transAxes, fontsize=16, color='#6c757d')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
        
        # Force garbage collection
        gc.collect()
        self.update_memory_usage()
        messagebox.showinfo("Data Cleared", "Data has been cleared from memory")

    def update_memory_usage(self):
        """Update memory usage display"""
        # This is a simplified representation
        if self.data is not None:
            memory_usage = self.data.memory_usage(deep=True).sum() / 1024 / 1024
            self.memory_label.config(text=f"Memory: {memory_usage:.2f} MB")
        elif self.dask_data is not None:
            self.memory_label.config(text=f"Memory: Large dataset (using Dask)")
        else:
            self.memory_label.config(text="Memory: 0 MB")

    def recommend_chart_type(self):
        if self.data is None and self.dask_data is None:
            self.recommend_label.config(text="💡 Load data for smart recommendations")
            return

        x_col = self.x_column.get()
        y_col = self.y_column.get()
        
        if not x_col or not y_col:
            self.recommend_label.config(text="💡 Select both axes for recommendations")
            return

        try:
            if self.data is not None:
                x_series = self.data[x_col]
                y_series = self.data[y_col]
            else:
                # For dask, we need to compute a sample
                sample = self.dask_data.sample(frac=0.01).compute()
                x_series = sample[x_col]
                y_series = sample[y_col]

            x_is_num = is_numeric_dtype(x_series)
            y_is_num = is_numeric_dtype(y_series)
            x_is_date = is_datetime64_any_dtype(x_series)
            x_unique = x_series.nunique() if not x_is_num else None
            y_cardinality = y_series.nunique()
            total_points = len(x_series)

            recommendation = "bar"
            reason = ""
            
            if x_is_date and y_is_num:
                recommendation = "line"
                reason = "Time series data detected"
            elif not x_is_num and y_is_num:
                if x_unique <= 7:
                    if total_points <= 50:
                        recommendation = "pie" if x_unique <= 5 else "bar"
                        reason = f"{x_unique} categories, good for comparison"
                    else:
                        recommendation = "bar"
                        reason = f"{x_unique} categories (too many points for pie)"
                else:
                    recommendation = "bar"
                    reason = f"Many categories ({x_unique}), best for comparisons"
            elif x_is_num and y_is_num:
                if x_series.is_monotonic_increasing:
                    recommendation = "line"
                    reason = "Ordered numerical data"
                else:
                    recommendation = "scatter"
                    reason = "Two numerical variables, shows correlation"
            elif x_is_num and not y_is_num:
                recommendation = "box"
                reason = "Numerical X with categorical Y"
            elif y_cardinality <= 5 and x_is_num:
                recommendation = "box"
                reason = "Limited Y categories, shows distribution"
            elif y_is_num and x_unique > 20:
                recommendation = "histogram"
                reason = "Many unique X values, show distribution"
            else:
                recommendation = "bar"
                reason = "Default recommendation"

            # Special overrides
            if recommendation == "pie" and total_points > 100:
                recommendation = "bar"
                reason = "Too many points for pie chart (use for <100 points)"
                
            if recommendation == "histogram" and not y_is_num:
                recommendation = "bar"
                reason = "Histogram requires numerical Y-axis"

            self.chart_type.set(recommendation)
            rec_text = f"🎯 Recommended: {recommendation.capitalize()} Chart\n💡 {reason}"
            self.recommend_label.config(text=rec_text)
        except KeyError:
            return

    def generate_chart(self):
        if self.data is None and self.dask_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        x_col = self.x_column.get()
        y_col = self.y_column.get()

        if not x_col or not y_col:
            messagebox.showwarning("Warning", "Please select X and Y columns")
            return

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        chart_type = self.chart_type.get()
        title = self.chart_title.get()

        try:
            colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14', '#20c997', '#6c757d']
            
            if self.data is not None:
                data = self.data
            else:
                # For dask, sample the data
                sample_size = min(5000, len(self.dask_data))
                data = self.dask_data.sample(sample_size).compute()
            
            # Handle large datasets with optimized plotting
            if chart_type == "bar":
                # For large datasets, sample or aggregate
                if len(data) > 10000:
                    sampled = data.sample(min(1000, len(data)))
                    sampled.plot(kind='bar', x=x_col, y=y_col, ax=self.ax, 
                                legend=self.legend.get(), color=colors[0])
                else:
                    data.plot(kind='bar', x=x_col, y=y_col, ax=self.ax, 
                            legend=self.legend.get(), color=colors[0])
            elif chart_type == "line":
                # For large time series, downsample
                if len(data) > 10000 and is_datetime64_any_dtype(data[x_col]):
                    data = data.set_index(x_col).resample('D').mean().reset_index()
                data.plot(kind='line', x=x_col, y=y_col, ax=self.ax, 
                        legend=self.legend.get(), color=colors[0], linewidth=2)
            elif chart_type == "scatter":
                # Sample for large datasets
                if len(data) > 5000:
                    sampled = data.sample(min(2000, len(data)))
                    sampled.plot(kind='scatter', x=x_col, y=y_col, ax=self.ax, 
                                color=colors[0], alpha=0.7)
                else:
                    data.plot(kind='scatter', x=x_col, y=y_col, ax=self.ax, 
                            color=colors[0], alpha=0.7)
            elif chart_type == "histogram":
                # For large datasets, use more bins
                bins = 50 if len(data) > 10000 else 20
                data[y_col].plot(kind='hist', ax=self.ax, legend=self.legend.get(), 
                                color=colors[0], alpha=0.8, bins=bins)
                self.ax.set_xlabel(y_col)
            elif chart_type == "box":
                # For large datasets, use sampled version
                if len(data) > 10000:
                    sampled = data.sample(min(5000, len(data)))
                    sampled[y_col].plot(kind='box', ax=self.ax, color=colors[0])
                else:
                    data[y_col].plot(kind='box', ax=self.ax, color=colors[0])
            elif chart_type == "pie":
                # Only show top categories for large datasets
                top_categories = 10
                if len(data) > 1000:
                    top_categories = 8
                pie_data = data.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(top_categories)
                pie_data.plot(kind='pie', ax=self.ax, autopct='%1.1f%%', colors=colors)

            self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            if chart_type not in ["pie", "box"]:
                self.ax.set_xlabel(x_col, fontsize=12)
                self.ax.set_ylabel(y_col, fontsize=12)

            self.ax.grid(self.grid_lines.get(), alpha=0.3)
            self.figure.tight_layout()
            self.canvas.draw()

            # Log visualization
            if self.user_id:
                try:
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO visualizations (user_id, chart_type, created_at) VALUES (?, ?, ?)",
                             (self.user_id, chart_type, datetime.now()))
                    conn.commit()
                except Exception as e:
                    print("Error logging visualization:", e)
                finally:
                    conn.close()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate chart: {str(e)}")

    def export_chart(self):
        if (self.data is None and self.dask_data is None) or self.ax is None:
            messagebox.showwarning("Warning", "No chart to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Chart",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf")]
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
            messagebox.showinfo("Success", "Chart exported successfully! 🎉")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart: {str(e)}")

    def save_work(self):
        if self.data is None and self.dask_data is None:
            messagebox.showwarning("Warning", "No work to save")
            return
            
        # Ask for a name for this saved work
        name = simpledialog.askstring("Save Work", "Enter a name for this saved work:")
        if not name:
            return

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            # Save the current state
            c.execute('''INSERT INTO saved_work 
                        (user_id, name, file_path, x_column, y_column, chart_type, 
                         chart_title, show_grid, show_legend, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (self.user_id, 
                      name,
                      self.current_file_path,
                      self.x_column.get(),
                      self.y_column.get(),
                      self.chart_type.get(),
                      self.chart_title.get(),
                      int(self.grid_lines.get()),
                      int(self.legend.get()),
                      datetime.now()))
            conn.commit()
            messagebox.showinfo("Success", "Work saved successfully! 💾")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save work: {str(e)}")
        finally:
            conn.close()

    def load_saved_work(self):
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT id, name, file_path, x_column, y_column, chart_type, chart_title, show_grid, show_legend FROM saved_work WHERE user_id=?", (self.user_id,))
            saved_items = c.fetchall()
            conn.close()
            
            if not saved_items:
                messagebox.showinfo("No Saved Work", "You have no saved work to load")
                return
                
            # Create a dialog to select saved work
            dialog = tk.Toplevel(self.root)
            dialog.title("Load Saved Work")
            dialog.geometry("500x400")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Title
            title_frame = ttk.Frame(dialog, padding=10)
            title_frame.pack(fill=tk.X)
            ttk.Label(title_frame, text="Select Saved Work to Load", font=('Segoe UI', 14, 'bold')).pack()
            
            # List of saved work
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.saved_work_list = ttk.Treeview(list_frame, columns=('name', 'date'), show='headings', yscrollcommand=scrollbar.set)
            self.saved_work_list.heading('name', text='Name')
            self.saved_work_list.heading('date', text='Date')
            self.saved_work_list.column('name', width=250)
            self.saved_work_list.column('date', width=150)
            
            for item in saved_items:
                self.saved_work_list.insert('', 'end', values=(item[1], item[0]))
                
            self.saved_work_list.pack(fill=tk.BOTH, expand=True)
            scrollbar.config(command=self.saved_work_list.yview)
            
            # Buttons
            btn_frame = ttk.Frame(dialog, padding=10)
            btn_frame.pack(fill=tk.X)
            
            load_btn = ttk.Button(btn_frame, text="Load Selected", command=lambda: self.load_selected_work(saved_items, dialog))
            load_btn.pack(side=tk.LEFT, padx=5)
            
            delete_btn = ttk.Button(btn_frame, text="Delete", command=lambda: self.delete_saved_work(saved_items, dialog))
            delete_btn.pack(side=tk.LEFT, padx=5)
            
            close_btn = ttk.Button(btn_frame, text="Cancel", command=dialog.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load saved work: {str(e)}")

    def load_selected_work(self, saved_items, dialog):
        selected = self.saved_work_list.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a saved work to load")
            return
            
        item_id = self.saved_work_list.item(selected[0])['values'][1]
        work = next((item for item in saved_items if item[0] == item_id), None)
        
        if work:
            # Extract saved work details
            _, name, file_path, x_col, y_col, chart_type, chart_title, show_grid, show_legend = work
            
            # Set UI values
            self.chart_type.set(chart_type)
            self.chart_title.set(chart_title)
            self.grid_lines.set(bool(show_grid))
            self.legend.set(bool(show_legend))
            
            # Try to load file
            if file_path and os.path.exists(file_path):
                # Start loading in a separate thread
                threading.Thread(target=self.load_file_async, args=(file_path,)).start()
                
                # Set column selections
                self.x_column.set(x_col)
                self.y_column.set(y_col)
            else:
                messagebox.showinfo("File Not Found", 
                                   f"The data file was not found:\n{file_path}")
            
            dialog.destroy()

    def delete_saved_work(self, saved_items, dialog):
        selected = self.saved_work_list.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a saved work to delete")
            return
            
        item_id = self.saved_work_list.item(selected[0])['values'][1]
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("DELETE FROM saved_work WHERE id=?", (item_id,))
            conn.commit()
            conn.close()
            
            # Refresh the list
            self.saved_work_list.delete(selected[0])
            messagebox.showinfo("Success", "Saved work deleted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete saved work: {str(e)}")

    def generate_admin_report(self):
        try:
            conn = sqlite3.connect('users.db')
            df_users = pd.read_sql("SELECT * FROM users", conn)
            df_viz = pd.read_sql("SELECT * FROM visualizations", conn)
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
            return

        report_window = tk.Toplevel()
        report_window.title("Admin Dashboard Report")
        report_window.geometry("1200x700")
        report_window.configure(bg="#f8f9fa")
        
        # Header
        header_frame = ttk.Frame(report_window)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = ttk.Label(header_frame, text="📊 Admin Dashboard Report", 
                               font=('Segoe UI', 18, 'bold'))
        title_label.pack(anchor='w')
        
        subtitle_label = ttk.Label(header_frame, text="Platform usage statistics and insights", 
                                  font=('Segoe UI', 11), foreground='#6c757d')
        subtitle_label.pack(anchor='w', pady=(5, 0))

        # Content area
        content_frame = ttk.Frame(report_window)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Statistics cards
        stats_frame = ttk.Frame(content_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        # User stats card
        user_stats_card = ModernCard(stats_frame, title="👥 User Statistics")
        user_stats_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        total_users = len(df_users)
        admin_users = len(df_users[df_users['role'] == 'admin'])
        regular_users = len(df_users[df_users['role'] == 'user'])
        
        ttk.Label(user_stats_card.content_frame, text=f"Total Users: {total_users}", 
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(user_stats_card.content_frame, text=f"👑 Admins: {admin_users}", 
                 font=('Segoe UI', 11)).pack(anchor='w', pady=2)
        ttk.Label(user_stats_card.content_frame, text=f"👤 Regular Users: {regular_users}", 
                 font=('Segoe UI', 11)).pack(anchor='w', pady=2)
        
        # Visualization stats card
        viz_stats_card = ModernCard(stats_frame, title="📈 Visualization Statistics")
        viz_stats_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        total_viz = len(df_viz)
        if total_viz > 0:
            most_popular = df_viz['chart_type'].value_counts().index[0]
            ttk.Label(viz_stats_card.content_frame, text=f"Total Visualizations: {total_viz}", 
                     font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=2)
            ttk.Label(viz_stats_card.content_frame, text=f"📊 Most Popular: {most_popular.title()}", 
                     font=('Segoe UI', 11)).pack(anchor='w', pady=2)
        else:
            ttk.Label(viz_stats_card.content_frame, text="No visualizations yet", 
                     font=('Segoe UI', 11), foreground='#6c757d').pack(anchor='w', pady=2)

        # Charts area
        charts_frame = ttk.Frame(content_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # User distribution chart
        user_chart_card = ModernCard(charts_frame, title="User Role Distribution")
        user_chart_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        fig1 = plt.Figure(figsize=(5, 4), dpi=100)
        fig1.patch.set_facecolor('#ffffff')
        ax1 = fig1.add_subplot(111)
        
        if not df_users.empty:
            user_counts = df_users['role'].value_counts()
            colors = ['#007bff', '#28a745']
            user_counts.plot(kind='bar', ax=ax1, color=colors)
            ax1.set_title("User Roles", fontweight='bold')
            ax1.set_xlabel("Role")
            ax1.set_ylabel("Count")
            ax1.tick_params(axis='x', rotation=0)
        else:
            ax1.text(0.5, 0.5, 'No user data', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12, color='#6c757d')
        
        fig1.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig1, master=user_chart_card.content_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Visualization distribution chart
        viz_chart_card = ModernCard(charts_frame, title="Visualization Types")
        viz_chart_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        fig2 = plt.Figure(figsize=(5, 4), dpi=100)
        fig2.patch.set_facecolor('#ffffff')
        ax2 = fig2.add_subplot(111)
        
        if not df_viz.empty:
            viz_counts = df_viz['chart_type'].value_counts()
            colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14']
            viz_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors)
            ax2.set_title("Chart Types Distribution", fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No visualization data', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, color='#6c757d')
        
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=viz_chart_card.content_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Action buttons
        btn_frame = ttk.Frame(report_window)
        btn_frame.pack(pady=20)
        
        save_btn = ttk.Button(btn_frame, text="💾 Save Report", 
                             command=lambda: self.save_admin_report(fig1, fig2))
        save_btn.pack(side=tk.LEFT, padx=10, ipady=8)
        
        close_btn = ttk.Button(btn_frame, text="❌ Close", 
                              command=report_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=10, ipady=8)

    def save_admin_report(self, fig1, fig2):
        file_path = filedialog.asksaveasfilename(
            title="Save Admin Report",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf")]
        )

        if not file_path:
            return

        try:
            # Save both charts
            base_path = os.path.splitext(file_path)[0]
            ext = os.path.splitext(file_path)[1]
            
            fig1.savefig(f"{base_path}_users{ext}", dpi=300, bbox_inches='tight', facecolor='white')
            fig2.savefig(f"{base_path}_visualizations{ext}", dpi=300, bbox_inches='tight', facecolor='white')
            
            messagebox.showinfo("Success", "Admin reports saved successfully! 🎉")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save reports: {str(e)}")

def main():
    root = tk.Tk()
    LoginWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()