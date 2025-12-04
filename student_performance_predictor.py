import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set dark theme for matplotlib
plt.style.use('dark_background')
sns.set_style("darkgrid")

class AnimatedButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.default_bg = self.cget('background')
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
    def on_enter(self, e):
        self.config(background='#45a049')
        
    def on_leave(self, e):
        self.config(background=self.default_bg)

class StudentPerformancePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Student Performance Predictor - Advanced ML Application")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize variables
        self.df = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_trained = False
        self.target_column = "performance"
        
        # Configure style
        self.setup_styles()
        
        # Create GUI
        self.create_gui()
        
        # Generate sample data
        self.generate_sample_data()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors for dark theme
        self.style.configure('TFrame', background='#1e1e1e')
        self.style.configure('TLabel', background='#1e1e1e', foreground='white', font=('Arial', 10))
        self.style.configure('Title.TLabel', background='#1e1e1e', foreground='#4CAF50', font=('Arial', 16, 'bold'))
        self.style.configure('TButton', background='#4CAF50', foreground='white', font=('Arial', 10))
        self.style.map('TButton', background=[('active', '#45a049')])
        self.style.configure('TNotebook', background='#1e1e1e', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#262730', foreground='white', padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', '#4CAF50')])
        self.style.configure('TCombobox', fieldbackground='#262730', background='#262730', foreground='white')
        self.style.configure('TEntry', fieldbackground='#262730', foreground='white')
        self.style.configure('TScale', background='#1e1e1e')
        
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🎓 Student Performance Predictor", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="📊 Data Explorer")
        self.notebook.add(self.model_tab, text="🤖 Model Training")
        self.notebook.add(self.predict_tab, text="🔮 Prediction")
        self.notebook.add(self.analytics_tab, text="📈 Analytics")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_predict_tab()
        self.setup_analytics_tab()
        
    def setup_data_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.data_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Data source frame
        source_frame = ttk.LabelFrame(left_frame, text="Data Source", padding=10)
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(source_frame, text="Generate Sample Data", 
                  command=self.generate_sample_data).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Upload CSV File", 
                  command=self.upload_csv).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Export Data", 
                  command=self.export_data).pack(fill=tk.X, pady=2)
        
        # Data info frame
        info_frame = ttk.LabelFrame(left_frame, text="Dataset Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, width=30, bg='#262730', fg='white', 
                                font=('Consolas', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization controls
        viz_frame = ttk.LabelFrame(left_frame, text="Visualization Controls", padding=10)
        viz_frame.pack(fill=tk.X)
        
        ttk.Label(viz_frame, text="X-Axis:").pack(anchor=tk.W)
        self.x_axis_var = tk.StringVar()
        self.x_combo = ttk.Combobox(viz_frame, textvariable=self.x_axis_var)
        self.x_combo.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(viz_frame, text="Y-Axis:").pack(anchor=tk.W)
        self.y_axis_var = tk.StringVar()
        self.y_combo = ttk.Combobox(viz_frame, textvariable=self.y_axis_var)
        self.y_combo.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(viz_frame, text="Update Plot", 
                  command=self.update_scatter_plot).pack(fill=tk.X)
        
        # Right frame for data display and visualization
        right_frame = ttk.Frame(self.data_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Data display
        data_display_frame = ttk.LabelFrame(right_frame, text="Data Preview", padding=10)
        data_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for data display
        self.tree = ttk.Treeview(data_display_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Visualization frame
        viz_display_frame = ttk.LabelFrame(right_frame, text="Visualizations", padding=10)
        viz_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.viz_frame = ttk.Frame(viz_display_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
    def setup_model_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.model_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Model configuration
        config_frame = ttk.LabelFrame(left_frame, text="Model Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(config_frame, text="Model Type:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_var,
                                  values=["Random Forest", "Gradient Boosting", "Linear Regression", 
                                         "Ridge Regression", "Support Vector Machine", "Neural Network"])
        model_combo.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(config_frame, text="Test Size (%):").pack(anchor=tk.W)
        self.test_size_var = tk.IntVar(value=20)
        ttk.Scale(config_frame, from_=10, to=40, variable=self.test_size_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(config_frame, text="Target Column:").pack(anchor=tk.W)
        self.target_var = tk.StringVar(value="performance")
        self.target_combo = ttk.Combobox(config_frame, textvariable=self.target_var)
        self.target_combo.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(config_frame, text="🚀 Train Model", 
                  command=self.train_model).pack(fill=tk.X, pady=5)
        
        # Model metrics
        metrics_frame = ttk.LabelFrame(left_frame, text="Model Performance", padding=10)
        metrics_frame.pack(fill=tk.X)
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=30, bg='#262730', fg='white',
                                   font=('Consolas', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Right frame for visualizations
        right_frame = ttk.Frame(self.model_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Model visualizations notebook
        self.model_viz_notebook = ttk.Notebook(right_frame)
        self.model_viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        self.feature_importance_frame = ttk.Frame(self.model_viz_notebook)
        self.prediction_frame = ttk.Frame(self.model_viz_notebook)
        self.residual_frame = ttk.Frame(self.model_viz_notebook)
        
        self.model_viz_notebook.add(self.feature_importance_frame, text="Feature Importance")
        self.model_viz_notebook.add(self.prediction_frame, text="Predictions vs Actual")
        self.model_viz_notebook.add(self.residual_frame, text="Residuals")
        
    def setup_predict_tab(self):
        main_frame = ttk.Frame(self.predict_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Prediction input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Features", padding=15)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create input fields dynamically based on data
        self.input_vars = {}
        self.input_frame = ttk.Frame(input_frame)
        self.input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prediction result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=15)
        result_frame.pack(fill=tk.X)
        
        self.result_text = tk.Text(result_frame, height=6, bg='#262730', fg='white',
                                  font=('Arial', 12), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(main_frame, text="🔮 Make Prediction", 
                  command=self.make_prediction).pack(pady=10)
        
    def setup_analytics_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.analytics_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Analytics controls
        controls_frame = ttk.LabelFrame(left_frame, text="Analytics Controls", padding=10)
        controls_frame.pack(fill=tk.X)
        
        ttk.Label(controls_frame, text="Compare Feature:").pack(anchor=tk.W)
        self.compare_var = tk.StringVar()
        self.compare_combo = ttk.Combobox(controls_frame, textvariable=self.compare_var)
        self.compare_combo.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Update Analytics", 
                  command=self.update_analytics).pack(fill=tk.X)
        
        # Right frame for analytics visualizations
        right_frame = ttk.Frame(self.analytics_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.analytics_frame = ttk.Frame(right_frame)
        self.analytics_frame.pack(fill=tk.BOTH, expand=True)
        
    def generate_sample_data(self):
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'study_hours': np.random.normal(15, 5, n_samples).clip(5, 30),
            'attendance': np.random.normal(85, 10, n_samples).clip(60, 100),
            'previous_scores': np.random.normal(75, 15, n_samples).clip(40, 100),
            'extracurricular': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
            'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(4, 10),
            'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'internet_access': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'tutoring': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5]),
            'age': np.random.randint(15, 20, n_samples)
        }
        
        # Calculate performance score
        performance = (
            data['study_hours'] * 0.3 +
            data['attendance'] * 0.2 +
            data['previous_scores'] * 0.3 +
            data['extracurricular'] * 2 +
            data['sleep_hours'] * 3 +
            (data['internet_access'] * 5) +
            (data['tutoring'] * 5) +
            np.random.normal(0, 5, n_samples)
        )
        
        performance = (performance - performance.min()) / (performance.max() - performance.min()) * 60 + 40
        data['performance'] = performance.clip(0, 100)
        
        self.df = pd.DataFrame(data)
        self.update_data_display()
        self.update_comboboxes()
        self.update_input_fields()
        messagebox.showinfo("Success", "Sample data generated successfully!")
        
    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_data_display()
                self.update_comboboxes()
                self.update_input_fields()
                messagebox.showinfo("Success", f"Data loaded successfully! Shape: {self.df.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
                
    def export_data(self):
        if self.df is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                                    filetypes=[("CSV files", "*.csv")])
            if file_path:
                try:
                    self.df.to_csv(file_path, index=False)
                    messagebox.showinfo("Success", "Data exported successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export data: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No data to export!")
            
    def update_data_display(self):
        if self.df is not None:
            # Clear existing tree
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            # Set up columns
            self.tree["columns"] = list(self.df.columns)
            self.tree["show"] = "headings"
            
            for column in self.df.columns:
                self.tree.heading(column, text=column)
                self.tree.column(column, width=100)
                
            # Add data
            for _, row in self.df.head(20).iterrows():
                self.tree.insert("", tk.END, values=list(row))
                
            # Update info text
            info = f"Dataset Shape: {self.df.shape}\n\n"
            info += f"Columns: {len(self.df.columns)}\n"
            info += f"Rows: {len(self.df)}\n\n"
            info += "Data Types:\n"
            for col, dtype in self.df.dtypes.items():
                info += f"  {col}: {dtype}\n"
                
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            
    def update_comboboxes(self):
        if self.df is not None:
            columns = list(self.df.columns)
            self.x_combo['values'] = columns
            self.y_combo['values'] = columns
            self.target_combo['values'] = columns
            self.compare_combo['values'] = [col for col in columns if col != self.target_column]
            
            if columns:
                self.x_combo.set(columns[0])
                self.y_combo.set(columns[1] if len(columns) > 1 else columns[0])
                self.compare_combo.set(columns[0] if columns[0] != self.target_column else columns[1] if len(columns) > 1 else columns[0])
                
    def update_input_fields(self):
        if self.df is not None and self.feature_names is not None:
            # Clear existing input fields
            for widget in self.input_frame.winfo_children():
                widget.destroy()
                
            self.input_vars = {}
            
            # Create input fields for each feature
            for i, feature in enumerate(self.feature_names):
                frame = ttk.Frame(self.input_frame)
                frame.grid(row=i//2, column=i%2, sticky='ew', padx=5, pady=5)
                
                ttk.Label(frame, text=feature).pack(anchor=tk.W)
                
                if self.df[feature].dtype == 'object':
                    options = list(self.df[feature].unique())
                    var = tk.StringVar(value=options[0] if options else "")
                    combo = ttk.Combobox(frame, textvariable=var, values=options)
                    combo.pack(fill=tk.X)
                    self.input_vars[feature] = var
                else:
                    min_val = float(self.df[feature].min())
                    max_val = float(self.df[feature].max())
                    avg_val = float(self.df[feature].mean())
                    
                    var = tk.DoubleVar(value=avg_val)
                    scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, 
                                     orient=tk.HORIZONTAL)
                    scale.pack(fill=tk.X)
                    
                    value_label = ttk.Label(frame, text=f"Value: {avg_val:.2f}")
                    value_label.pack()
                    
                    # Update label when scale changes
                    def update_label(val, label=value_label, v=var):
                        label.config(text=f"Value: {v.get():.2f}")
                    
                    scale.configure(command=update_label)
                    self.input_vars[feature] = var
                    
            # Configure grid weights
            for i in range(2):
                self.input_frame.columnconfigure(i, weight=1)
                
    def update_scatter_plot(self):
        if self.df is not None and self.x_axis_var.get() and self.y_axis_var.get():
            try:
                # Clear previous plot
                for widget in self.viz_frame.winfo_children():
                    widget.destroy()
                    
                # Create figure
                fig = Figure(figsize=(8, 6), dpi=100)
                ax = fig.add_subplot(111)
                
                x_col = self.x_axis_var.get()
                y_col = self.y_axis_var.get()
                
                if self.df[x_col].dtype == 'object':
                    # Box plot for categorical x
                    data_to_plot = [self.df[self.df[x_col] == cat][y_col] for cat in self.df[x_col].unique()]
                    ax.boxplot(data_to_plot, labels=self.df[x_col].unique())
                    ax.set_xticklabels(self.df[x_col].unique(), rotation=45)
                else:
                    # Scatter plot for numerical
                    ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, color='#4CAF50')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    
                    # Add trendline
                    if len(self.df) > 1:
                        z = np.polyfit(self.df[x_col], self.df[y_col], 1)
                        p = np.poly1d(z)
                        ax.plot(self.df[x_col], p(self.df[x_col]), "r--", alpha=0.8)
                
                ax.set_title(f"{y_col} vs {x_col}")
                ax.grid(True, alpha=0.3)
                
                # Embed in tkinter
                canvas = FigureCanvasTkAgg(fig, self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create plot: {str(e)}")
                
    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            target_col = self.target_var.get()
            if target_col not in self.df.columns:
                messagebox.showerror("Error", f"Target column '{target_col}' not found in data!")
                return
                
            # Prepare data
            X = self.df.drop(columns=[target_col])
            y = self.df[target_col]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                
            # Split data
            test_size = self.test_size_var.get() / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Get model
            model_type = self.model_var.get()
            if model_type == "Random Forest":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "Gradient Boosting":
                self.model = GradientBoostingRegressor(random_state=42)
            elif model_type == "Linear Regression":
                self.model = LinearRegression()
            elif model_type == "Ridge Regression":
                self.model = Ridge(alpha=1.0)
            elif model_type == "Support Vector Machine":
                self.model = SVR(kernel='rbf')
            elif model_type == "Neural Network":
                self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            self.model_trained = True
            self.target_column = target_col
            
            # Update metrics display
            metrics_text = f"Model: {model_type}\n"
            metrics_text += f"R² Score: {r2:.4f}\n"
            metrics_text += f"MSE: {mse:.4f}\n"
            metrics_text += f"RMSE: {rmse:.4f}\n"
            metrics_text += f"Test Size: {test_size*100}%\n"
            metrics_text += f"Features: {len(self.feature_names)}\n"
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(1.0, metrics_text)
            
            # Update input fields
            self.update_input_fields()
            
            # Create visualizations
            self.create_model_visualizations(X_test, y_test, y_pred)
            
            messagebox.showinfo("Success", "Model trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            
    def create_model_visualizations(self, X_test, y_test, y_pred):
        # Feature Importance
        self.create_feature_importance_plot()
        
        # Predictions vs Actual
        self.create_prediction_plot(y_test, y_pred)
        
        # Residuals
        self.create_residual_plot(y_test, y_pred)
        
    def create_feature_importance_plot(self):
        if hasattr(self.model, 'feature_importances_'):
            # Clear frame
            for widget in self.feature_importance_frame.winfo_children():
                widget.destroy()
                
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create figure
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            features = [self.feature_names[i] for i in indices]
            y_pos = np.arange(len(features))
            
            ax.barh(y_pos, importances[indices], color='#4CAF50')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3)
            
            # Embed
            canvas = FigureCanvasTkAgg(fig, self.feature_importance_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
    def create_prediction_plot(self, y_test, y_pred):
        # Clear frame
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
            
        # Create figure
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.scatter(y_test, y_pred, alpha=0.6, color='#2196F3')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.grid(True, alpha=0.3)
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, self.prediction_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_residual_plot(self, y_test, y_pred):
        # Clear frame
        for widget in self.residual_frame.winfo_children():
            widget.destroy()
            
        # Create figure
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, color='#FF9800')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, self.residual_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def make_prediction(self):
        if not self.model_trained:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        try:
            # Prepare input data
            input_data = {}
            for feature, var in self.input_vars.items():
                input_data[feature] = var.get()
                
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            categorical_cols = input_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in self.df.columns:
                    le = LabelEncoder()
                    le.fit(self.df[col])
                    input_df[col] = le.transform(input_df[col])
                    
            # Ensure correct column order
            input_df = input_df[self.feature_names]
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Display result
            result_text = f"🎯 Predicted Performance Score: {prediction:.2f}\n\n"
            
            if prediction >= 90:
                result_text += "📊 Performance: Excellent! 🎉\n"
                result_text += "The student is performing exceptionally well."
            elif prediction >= 80:
                result_text += "📊 Performance: Very Good! 👍\n"
                result_text += "The student is performing very well."
            elif prediction >= 70:
                result_text += "📊 Performance: Good 📈\n"
                result_text += "Good performance with room for improvement."
            elif prediction >= 60:
                result_text += "📊 Performance: Average ⚠️\n"
                result_text += "Average performance, needs attention."
            else:
                result_text += "📊 Performance: Below Average 🚨\n"
                result_text += "Requires immediate intervention and support."
                
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
            
    def update_analytics(self):
        if self.df is None:
            return
            
        try:
            # Clear previous analytics
            for widget in self.analytics_frame.winfo_children():
                widget.destroy()
                
            compare_feature = self.compare_var.get()
            if not compare_feature:
                return
                
            # Create subplots
            fig = Figure(figsize=(12, 8), dpi=100)
            
            if self.df[compare_feature].dtype == 'object' or self.df[compare_feature].nunique() < 10:
                # Categorical feature - use box plot
                ax1 = fig.add_subplot(221)
                data_to_plot = [self.df[self.df[compare_feature] == cat][self.target_column] 
                               for cat in self.df[compare_feature].unique()]
                ax1.boxplot(data_to_plot, labels=self.df[compare_feature].unique())
                ax1.set_xticklabels(self.df[compare_feature].unique(), rotation=45)
                ax1.set_title(f'Performance by {compare_feature}')
                ax1.set_ylabel('Performance Score')
                ax1.grid(True, alpha=0.3)
                
                # Performance distribution by category
                ax2 = fig.add_subplot(222)
                performance_by_cat = self.df.groupby(compare_feature)[self.target_column].mean()
                performance_by_cat.plot(kind='bar', ax=ax2, color='#4CAF50')
                ax2.set_title(f'Average Performance by {compare_feature}')
                ax2.set_ylabel('Average Performance Score')
                ax2.grid(True, alpha=0.3)
                
            else:
                # Numerical feature - use scatter plot
                ax1 = fig.add_subplot(221)
                ax1.scatter(self.df[compare_feature], self.df[self.target_column], 
                           alpha=0.6, color='#2196F3')
                ax1.set_xlabel(compare_feature)
                ax1.set_ylabel('Performance Score')
                ax1.set_title(f'Performance vs {compare_feature}')
                ax1.grid(True, alpha=0.3)
                
                # Add trendline
                if len(self.df) > 1:
                    z = np.polyfit(self.df[compare_feature], self.df[self.target_column], 1)
                    p = np.poly1d(z)
                    ax1.plot(self.df[compare_feature], p(self.df[compare_feature]), "r--", alpha=0.8)
            
            # Performance distribution
            ax3 = fig.add_subplot(223)
            ax3.hist(self.df[self.target_column], bins=20, color='#FF9800', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Performance Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Performance Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Correlation heatmap (numerical columns only)
            ax4 = fig.add_subplot(224)
            numerical_df = self.df.select_dtypes(include=[np.number])
            if len(numerical_df.columns) > 1:
                corr_matrix = numerical_df.corr()
                im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax4.set_xticks(range(len(corr_matrix.columns)))
                ax4.set_yticks(range(len(corr_matrix.columns)))
                ax4.set_xticklabels(corr_matrix.columns, rotation=45)
                ax4.set_yticklabels(corr_matrix.columns)
                ax4.set_title('Correlation Heatmap')
                
                # Add correlation values
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                ha='center', va='center', color='white', fontsize=8)
                
                fig.colorbar(im, ax=ax4)
            
            fig.tight_layout()
            
            # Embed
            canvas = FigureCanvasTkAgg(fig, self.analytics_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update analytics: {str(e)}")

def main():
    root = tk.Tk()
    app = StudentPerformancePredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()