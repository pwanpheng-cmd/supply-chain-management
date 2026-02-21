import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
st.set_page_config(
    page_title="Supply Chain Management Pro",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .upload-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    .module-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .alert-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Core Classes ====================

class SCMSystem:
    """Supply Chain Management System - Main Controller"""
    
    def __init__(self):
        self.sales_data = None
        self.stock_data = None
        self.promotion_data = None
        self.po_data = None
    
    def load_sales_data(self, file):
        """Load sales data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Validate required columns
            required_cols = ['date', 'product_id', 'sales_qty']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                return False, f"Missing columns: {', '.join(missing)}"
            
            self.sales_data = df.sort_values('date')
            return True, f"✅ Loaded {len(df)} sales records"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def load_stock_data(self, file):
        """Load stock data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            required_cols = ['date', 'product_id', 'ending_stock']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                return False, f"Missing columns: {', '.join(missing)}"
            
            self.stock_data = df.sort_values('date')
            return True, f"✅ Loaded {len(df)} stock records"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def load_promotion_data(self, file):
        """Load promotion data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            if 'start_date' in df.columns:
                df['start_date'] = pd.to_datetime(df['start_date'])
            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'])
            
            self.promotion_data = df
            return True, f"✅ Loaded {len(df)} promotions"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def load_po_data(self, file):
        """Load purchase order data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            if 'po_date' in df.columns:
                df['po_date'] = pd.to_datetime(df['po_date'])
            if 'expected_delivery' in df.columns:
                df['expected_delivery'] = pd.to_datetime(df['expected_delivery'])
            if 'actual_delivery' in df.columns:
                df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])
            
            self.po_data = df
            return True, f"✅ Loaded {len(df)} purchase orders"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
        
    def create_sample_sales_data(self, days=90, products=3):
        """สร้างข้อมูลยอดขายตัวอย่าง"""
        np.random.seed(42)
        all_data = []
        
        product_info = {
            'P001': {'name': 'Milk UHT 1L', 'base': 1000, 'category': 'Dairy'},
            'P002': {'name': 'Orange Juice 1L', 'base': 1200, 'category': 'Beverage'},
            'P003': {'name': 'Yogurt 150g', 'base': 800, 'category': 'Dairy'}
        }
        
        for i in range(products):
            product_id = f'P00{i+1}'
            info = product_info.get(product_id, {'name': f'Product {i+1}', 'base': 1000, 'category': 'General'})
            
            dates = [datetime.now().date() - timedelta(days=j) for j in range(days, 0, -1)]
            
            base = info['base']
            trend = np.linspace(0, 200, days)
            seasonality = 150 * np.sin(np.linspace(0, 6*np.pi, days))
            noise = np.random.normal(0, 50, days)
            
            sales = base + trend + seasonality + noise
            sales = np.maximum(sales, 0)
            
            unit_price = 45.0 if i == 0 else (38.0 if i == 1 else 12.0)
            
            df_product = pd.DataFrame({
                'date': dates,
                'product_id': product_id,
                'product_name': info['name'],
                'sales_qty': sales.astype(int),
                'unit_price': unit_price,
                'total_amount': (sales * unit_price).astype(int),
                'category': info['category'],
                'channel': np.random.choice(['CJ', 'Express', 'Access'], days)
            })
            
            all_data.append(df_product)
        
        self.sales_data = pd.concat(all_data, ignore_index=True)
        return True
    
    def create_sample_stock_data(self, days=30):
        """สร้างข้อมูลสต็อกตัวอย่าง"""
        if self.sales_data is None:
            return False
        
        np.random.seed(43)
        all_stock = []
        
        products = self.sales_data['product_id'].unique()
        
        for product_id in products:
            product_sales = self.sales_data[self.sales_data['product_id'] == product_id].copy()
            dates = sorted(product_sales['date'].unique())[-days:]
            
            beginning_stock = 2000
            
            for date in dates:
                daily_sales = product_sales[product_sales['date'] == date]['sales_qty'].sum()
                
                goods_received = np.random.choice([0, 500, 1000, 1500], p=[0.7, 0.15, 0.1, 0.05])
                
                ending_stock = beginning_stock + goods_received - daily_sales
                ending_stock = max(0, ending_stock)
                
                all_stock.append({
                    'date': date,
                    'product_id': product_id,
                    'beginning_stock': int(beginning_stock),
                    'goods_received': int(goods_received),
                    'sales_out': int(daily_sales),
                    'adjustment': 0,
                    'ending_stock': int(ending_stock),
                    'unit_cost': 35.0 if product_id == 'P001' else (30.0 if product_id == 'P002' else 9.0),
                    'stock_value': int(ending_stock * (35.0 if product_id == 'P001' else (30.0 if product_id == 'P002' else 9.0)))
                })
                
                beginning_stock = ending_stock
        
        self.stock_data = pd.DataFrame(all_stock)
        return True
    
    def create_sample_promotion_data(self):
        """สร้างข้อมูลโปรโมชั่นตัวอย่าง"""
        promotions = [
            {
                'promotion_id': 'PROMO001',
                'promotion_name': 'New Year Sale',
                'product_id': 'P001',
                'start_date': datetime.now().date() - timedelta(days=60),
                'end_date': datetime.now().date() - timedelta(days=50),
                'discount_type': 'Percentage',
                'discount_value': 15.0,
                'target_sales': 5000,
                'actual_sales': 5200,
                'status': 'Completed'
            },
            {
                'promotion_id': 'PROMO002',
                'promotion_name': 'Valentine Special',
                'product_id': 'P002',
                'start_date': datetime.now().date() - timedelta(days=30),
                'end_date': datetime.now().date() - timedelta(days=20),
                'discount_type': 'Percentage',
                'discount_value': 20.0,
                'target_sales': 6000,
                'actual_sales': 6500,
                'status': 'Completed'
            },
            {
                'promotion_id': 'PROMO003',
                'promotion_name': 'Summer Flash Sale',
                'product_id': 'P003',
                'start_date': datetime.now().date() - timedelta(days=5),
                'end_date': datetime.now().date() + timedelta(days=5),
                'discount_type': 'Percentage',
                'discount_value': 25.0,
                'target_sales': 4000,
                'actual_sales': 2100,
                'status': 'Active'
            }
        ]
        
        self.promotion_data = pd.DataFrame(promotions)
        return True
    
    def create_sample_po_data(self):
        """สร้างข้อมูล Purchase Order ตัวอย่าง"""
        pos = []
        
        for i in range(10):
            po_date = datetime.now().date() - timedelta(days=np.random.randint(1, 60))
            product_id = np.random.choice(['P001', 'P002', 'P003'])
            qty = np.random.choice([500, 1000, 1500, 2000])
            unit_cost = 35.0 if product_id == 'P001' else (30.0 if product_id == 'P002' else 9.0)
            
            expected_delivery = po_date + timedelta(days=np.random.randint(7, 14))
            actual_delivery = expected_delivery + timedelta(days=np.random.randint(-2, 3)) if po_date < datetime.now().date() - timedelta(days=7) else None
            
            status = 'Received' if actual_delivery else ('In Transit' if po_date < datetime.now().date() - timedelta(days=3) else 'Pending')
            
            pos.append({
                'po_id': f'PO2025{i+1:04d}',
                'po_date': po_date,
                'product_id': product_id,
                'supplier_id': f'SUP00{np.random.randint(1, 4)}',
                'supplier_name': f'Supplier {np.random.randint(1, 4)}',
                'order_qty': qty,
                'unit_cost': unit_cost,
                'total_amount': qty * unit_cost,
                'expected_delivery': expected_delivery,
                'actual_delivery': actual_delivery,
                'status': status,
                'lead_time_days': (actual_delivery - po_date).days if actual_delivery else (expected_delivery - po_date).days
            })
        
        self.po_data = pd.DataFrame(pos)
        return True
    
    def calculate_stock_metrics(self, product_id):
        """คำนวณ Stock Metrics"""
        if self.stock_data is None or self.sales_data is None:
            return None
        
        product_stock = self.stock_data[self.stock_data['product_id'] == product_id].copy()
        product_sales = self.sales_data[self.sales_data['product_id'] == product_id].copy()
        
        current_stock = product_stock['ending_stock'].iloc[-1]
        avg_daily_sales = product_sales['sales_qty'].mean()
        days_on_hand = current_stock / avg_daily_sales if avg_daily_sales > 0 else 0
        
        last_30_days = product_stock.tail(30)
        total_sales = last_30_days['sales_out'].sum() if 'sales_out' in last_30_days.columns else 0
        avg_stock = last_30_days['ending_stock'].mean()
        stock_turnover = (total_sales / avg_stock) if avg_stock > 0 else 0
        
        stock_value = product_stock['stock_value'].iloc[-1] if 'stock_value' in product_stock.columns else 0
        
        return {
            'current_stock': int(current_stock),
            'avg_daily_sales': int(avg_daily_sales),
            'days_on_hand': round(days_on_hand, 1),
            'stock_turnover': round(stock_turnover, 2),
            'stock_value': int(stock_value),
            'status': 'Low' if days_on_hand < 7 else ('High' if days_on_hand > 30 else 'Normal')
        }
    
    def ai_forecast_sales(self, product_id, days=30, promotion_impact=0):
        """AI Sales Forecasting"""
        if self.sales_data is None:
            return None
        
        product_data = self.sales_data[self.sales_data['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        
        if len(product_data) < 14:
            return None
        
        ma_7 = product_data['sales_qty'].rolling(window=7).mean().iloc[-1]
        ma_30 = product_data['sales_qty'].rolling(window=30).mean().iloc[-1] if len(product_data) >= 30 else ma_7
        
        product_data['day_num'] = range(len(product_data))
        X = product_data[['day_num']].values
        y = product_data['sales_qty'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_day = product_data['day_num'].max()
        future_X = np.array([[last_day + i + 1] for i in range(days)])
        lr_predictions = model.predict(future_X)
        
        base_forecast = []
        for i in range(days):
            ma_pred = ma_7 if i < 7 else ma_30
            lr_pred = lr_predictions[i]
            combined = (ma_pred * 0.5 + lr_pred * 0.5)
            base_forecast.append(max(0, combined))
        
        final_forecast = np.array(base_forecast)
        if promotion_impact > 0:
            final_forecast = final_forecast * (1 + promotion_impact / 100)
        
        last_date = product_data['date'].max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast_sales': final_forecast.astype(int),
            'method': f'AI Hybrid (Promotion: {promotion_impact}%)'
        })

# ==================== Initialize System ====================

if 'scm' not in st.session_state:
    st.session_state.scm = SCMSystem()
    st.session_state.data_loaded = False

scm = st.session_state.scm

# ==================== Header ====================
st.markdown('<h1 class="main-header">🏭 Supply Chain Management System</h1>', unsafe_allow_html=True)
st.markdown("### Complete Integrated Solution: Sales | Inventory | Promotion | Supply Planning")
st.markdown("---")

# ==================== Sidebar ====================
st.sidebar.image("https://img.icons8.com/fluency/96/000000/supply-chain.png", width=100)
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Data Management Section
st.sidebar.subheader("📁 Data Management")

data_option = st.sidebar.radio(
    "Choose data source:",
    ["📤 Upload Files", "🎲 Use Sample Data"],
    label_visibility="collapsed"
)

if data_option == "🎲 Use Sample Data":
    if st.sidebar.button("🎲 Load Sample Data", type="primary", use_container_width=True):
        with st.spinner("Loading sample data..."):
            scm.create_sample_sales_data(90, 3)
            scm.create_sample_stock_data(30)
            scm.create_sample_promotion_data()
            scm.create_sample_po_data()
            st.session_state.data_loaded = True
        st.sidebar.success("✅ Sample data loaded!")
        st.rerun()

st.sidebar.markdown("---")

# Module Selection
st.sidebar.subheader("📊 Modules")
selected_module = st.sidebar.radio(
    "Select Module:",
    ["🏠 Dashboard", "📤 Upload Data", "📈 Sales Forecasting", "📦 Inventory Management", 
     "🎯 Promotion Planning", "📋 Supply Planning", "📊 Reports"],
    label_visibility="collapsed"
)

# ==================== Main Content ====================

# ==================== 📤 UPLOAD DATA MODULE ====================
if selected_module == "📤 Upload Data":
    st.header("📤 Upload Your Data")
    
    st.markdown("""
    Upload your CSV or Excel files to use real data in the system.
    Make sure your files have the required columns as shown in the templates.
    """)
    
    # Create tabs for different data types
    upload_tab1, upload_tab2, upload_tab3, upload_tab4 = st.tabs([
        "📈 Sales Data", "📦 Stock Data", "🎯 Promotion Data", "📋 Purchase Orders"
    ])
    
    # Sales Data Upload
    with upload_tab1:
        st.subheader("📈 Upload Sales Data")
        
        st.markdown("""
        **Required columns:** `date`, `product_id`, `sales_qty`
        
        **Optional columns:** `product_name`, `unit_price`, `total_amount`, `category`, `channel`
        """)
        
        sales_file = st.file_uploader(
            "Choose Sales Data File",
            type=['csv', 'xlsx', 'xls'],
            key='sales_upload',
            help="Upload CSV or Excel file with sales data"
        )
        
        if sales_file:
            success, message = scm.load_sales_data(sales_file)
            
            if success:
                st.success(message)
                st.session_state.data_loaded = True
                
                # Preview data
                st.markdown("#### 📋 Data Preview")
                st.dataframe(scm.sales_data.head(10), use_container_width=True)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(scm.sales_data))
                with col2:
                    st.metric("Products", len(scm.sales_data['product_id'].unique()))
                with col3:
                    st.metric("Date Range", f"{scm.sales_data['date'].min().date()} to {scm.sales_data['date'].max().date()}")
            else:
                st.error(message)
        
        # Download Template
        st.markdown("---")
        st.markdown("#### 📥 Download Template")
        
        template_sales = """date,product_id,product_name,sales_qty,unit_price,total_amount,category,channel
2025-01-01,P001,Milk UHT 1L,1050,45.00,47250,Dairy,CJ
2025-01-01,P002,Orange Juice 1L,1180,38.00,44840,Beverage,Express
2025-01-01,P003,Yogurt 150g,820,12.00,9840,Dairy,Access"""
        
        st.download_button(
            "📥 Download Sales Template (CSV)",
            template_sales,
            "sales_template.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Stock Data Upload
    with upload_tab2:
        st.subheader("📦 Upload Stock Data")
        
        st.markdown("""
        **Required columns:** `date`, `product_id`, `ending_stock`
        
        **Optional columns:** `beginning_stock`, `goods_received`, `sales_out`, `adjustment`, `unit_cost`, `stock_value`
        """)
        
        stock_file = st.file_uploader(
            "Choose Stock Data File",
            type=['csv', 'xlsx', 'xls'],
            key='stock_upload'
        )
        
        if stock_file:
            success, message = scm.load_stock_data(stock_file)
            
            if success:
                st.success(message)
                st.markdown("#### 📋 Data Preview")
                st.dataframe(scm.stock_data.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(scm.stock_data))
                with col2:
                    st.metric("Products", len(scm.stock_data['product_id'].unique()))
            else:
                st.error(message)
        
        st.markdown("---")
        template_stock = """date,product_id,beginning_stock,goods_received,sales_out,adjustment,ending_stock,unit_cost,stock_value
2025-01-01,P001,2000,0,1050,0,950,35.00,33250
2025-01-01,P002,2500,500,1180,0,1820,30.00,54600
2025-01-01,P003,1800,0,820,0,980,9.00,8820"""
        
        st.download_button(
            "📥 Download Stock Template (CSV)",
            template_stock,
            "stock_template.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Promotion Data Upload
    with upload_tab3:
        st.subheader("🎯 Upload Promotion Data")
        
        st.markdown("""
        **Columns:** `promotion_id`, `promotion_name`, `product_id`, `start_date`, `end_date`, `discount_type`, `discount_value`, `target_sales`, `actual_sales`, `status`
        """)
        
        promo_file = st.file_uploader(
            "Choose Promotion Data File",
            type=['csv', 'xlsx', 'xls'],
            key='promo_upload'
        )
        
        if promo_file:
            success, message = scm.load_promotion_data(promo_file)
            
            if success:
                st.success(message)
                st.dataframe(scm.promotion_data, use_container_width=True)
            else:
                st.error(message)
        
        st.markdown("---")
        template_promo = """promotion_id,promotion_name,product_id,start_date,end_date,discount_type,discount_value,target_sales,actual_sales,status
PROMO001,New Year Sale,P001,2025-01-01,2025-01-10,Percentage,15,5000,5200,Completed
PROMO002,Valentine Special,P002,2025-02-10,2025-02-14,Percentage,20,6000,6500,Completed"""
        
        st.download_button(
            "📥 Download Promotion Template (CSV)",
            template_promo,
            "promotion_template.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Purchase Order Upload
    with upload_tab4:
        st.subheader("📋 Upload Purchase Order Data")
        
        st.markdown("""
        **Columns:** `po_id`, `po_date`, `product_id`, `supplier_id`, `supplier_name`, `order_qty`, `unit_cost`, `total_amount`, `expected_delivery`, `actual_delivery`, `status`, `lead_time_days`
        """)
        
        po_file = st.file_uploader(
            "Choose PO Data File",
            type=['csv', 'xlsx', 'xls'],
            key='po_upload'
        )
        
        if po_file:
            success, message = scm.load_po_data(po_file)
            
            if success:
                st.success(message)
                st.dataframe(scm.po_data, use_container_width=True)
            else:
                st.error(message)
        
        st.markdown("---")
        template_po = """po_id,po_date,product_id,supplier_id,supplier_name,order_qty,unit_cost,total_amount,expected_delivery,actual_delivery,status,lead_time_days
PO20250001,2025-01-05,P001,SUP001,Supplier ABC,1000,35.00,35000,2025-01-12,2025-01-13,Received,8
PO20250002,2025-01-10,P002,SUP002,Supplier XYZ,1500,30.00,45000,2025-01-24,2025-01-23,Received,13"""
        
        st.download_button(
            "📥 Download PO Template (CSV)",
            template_po,
            "po_template.csv",
            "text/csv",
            use_container_width=True
        )

# ==================== Check if data is loaded ====================
elif not st.session_state.data_loaded:
    st.info("👆 Please load data first! Choose 'Upload Data' or 'Use Sample Data' from the sidebar.")
    st.stop()

# ==================== 🏠 DASHBOARD ====================
elif selected_module == "🏠 Dashboard":
    st.header("🏠 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if scm.sales_data is not None:
            total_sales = scm.sales_data['total_amount'].sum() if 'total_amount' in scm.sales_data.columns else scm.sales_data['sales_qty'].sum()
            st.metric("💰 Total Sales", f"฿{total_sales:,.0f}")
    
    with col2:
        if scm.stock_data is not None:
            total_stock_value = scm.stock_data['stock_value'].iloc[-3:].sum() if 'stock_value' in scm.stock_data.columns else 0
            st.metric("📦 Stock Value", f"฿{total_stock_value:,.0f}")
    
    with col3:
        if scm.promotion_data is not None:
            active_promos = len(scm.promotion_data[scm.promotion_data['status'] == 'Active'])
            st.metric("🎯 Active Promotions", active_promos)
    
    with col4:
        if scm.po_data is not None:
            pending_pos = len(scm.po_data[scm.po_data['status'].isin(['Pending', 'In Transit'])])
            st.metric("📋 Pending POs", pending_pos)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if scm.sales_data is not None:
            st.subheader("📈 Sales Trend (Last 30 Days)")
            last_30 = scm.sales_data.tail(300)
            
            if 'total_amount' in last_30.columns:
                daily_sales = last_30.groupby('date')['total_amount'].sum().reset_index()
                y_col = 'total_amount'
                y_label = 'Revenue (฿)'
            else:
                daily_sales = last_30.groupby('date')['sales_qty'].sum().reset_index()
                y_col = 'sales_qty'
                y_label = 'Sales Quantity'
            
            fig = px.area(daily_sales, x='date', y=y_col,
                         title='Daily Sales',
                         labels={y_col: y_label, 'date': 'Date'})
            fig.update_traces(fillcolor='rgba(102, 126, 234, 0.3)', line_color='#667eea')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if scm.stock_data is not None:
            st.subheader("📦 Stock Status by Product")
            latest_stock = scm.stock_data.groupby('product_id').last().reset_index()
            
            fig = px.bar(latest_stock, x='product_id', y='ending_stock',
                        title='Current Stock Levels',
                        labels={'ending_stock': 'Stock Quantity', 'product_id': 'Product'},
                        color='ending_stock',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# ==================== 📈 SALES FORECASTING ====================
elif selected_module == "📈 Sales Forecasting":
    st.header("📈 Sales Forecasting")
    
    if scm.sales_data is not None:
        products = scm.sales_data['product_id'].unique()
        selected_product = st.selectbox("Select Product", products)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_days = st.slider("Forecast Days", 7, 90, 30)
        
        with col2:
            promotion_impact = st.slider("Promotion Impact (%)", 0, 100, 0, 5)
        
        with col3:
            if st.button("🤖 Generate Forecast", type="primary"):
                with st.spinner("AI is forecasting..."):
                    forecast = scm.ai_forecast_sales(selected_product, forecast_days, promotion_impact)
                    if forecast is not None:
                        st.session_state.forecast = forecast
                        st.success("✅ Forecast generated!")
                    else:
                        st.error("Not enough data for forecasting")
        
        if 'forecast' in st.session_state:
            forecast = st.session_state.forecast
            product_data = scm.sales_data[scm.sales_data['product_id'] == selected_product].copy()
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Forecast", f"{forecast['forecast_sales'].mean():,.0f} units/day")
            
            with col2:
                st.metric("Total Forecast", f"{forecast['forecast_sales'].sum():,.0f} units")
            
            with col3:
                current_avg = product_data['sales_qty'].mean()
                forecast_avg = forecast['forecast_sales'].mean()
                change = ((forecast_avg / current_avg - 1) * 100)
                st.metric("Change vs Current", f"{change:+.1f}%")
            
            with col4:
                st.metric("Forecast Period", f"{forecast_days} days")
            
            st.subheader("📊 Sales Forecast Chart")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=product_data['date'].tail(30),
                y=product_data['sales_qty'].tail(30),
                name='Historical Sales',
                mode='lines+markers',
                line=dict(color='#667eea', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['forecast_sales'],
                name='AI Forecast',
                mode='lines+markers',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            fig.update_layout(
                title=f'Sales Forecast - {selected_product}',
                xaxis_title='Date',
                yaxis_title='Sales Quantity',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View Forecast Data"):
                st.dataframe(forecast, use_container_width=True)

# ==================== 📦 INVENTORY MANAGEMENT ====================
elif selected_module == "📦 Inventory Management":
    st.header("📦 Inventory Management")
    
    if scm.stock_data is not None and scm.sales_data is not None:
        products = scm.stock_data['product_id'].unique()
        selected_product = st.selectbox("Select Product", products)
        
        metrics = scm.calculate_stock_metrics(selected_product)
        
        if metrics:
            st.subheader(f"📊 Stock Metrics - {selected_product}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Stock", f"{metrics['current_stock']:,} units")
            
            with col2:
                st.metric("Avg Daily Sales", f"{metrics['avg_daily_sales']:,} units")
            
            with col3:
                delta_color = "normal" if metrics['status'] == 'Normal' else ("inverse" if metrics['status'] == 'Low' else "off")
                st.metric("Days on Hand", f"{metrics['days_on_hand']:.1f} days", 
                         delta=metrics['status'], delta_color=delta_color)
            
            with col4:
                st.metric("Stock Turnover", f"{metrics['stock_turnover']:.2f}x")
            
            with col5:
                st.metric("Stock Value", f"฿{metrics['stock_value']:,}")
            
            st.markdown("---")
            
            st.subheader("📈 Stock Movement (Last 30 Days)")
            
            product_stock = scm.stock_data[scm.stock_data['product_id'] == selected_product].tail(30)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=product_stock['date'],
                y=product_stock['ending_stock'],
                name='Ending Stock',
                fill='tozeroy',
                line=dict(color='#667eea', width=2)
            ))
            
            if 'goods_received' in product_stock.columns:
                fig.add_trace(go.Bar(
                    x=product_stock['date'],
                    y=product_stock['goods_received'],
                    name='Goods Received',
                    marker_color='#28a745'
                ))
            
            if 'sales_out' in product_stock.columns:
                fig.add_trace(go.Bar(
                    x=product_stock['date'],
                    y=-product_stock['sales_out'],
                    name='Sales Out',
                    marker_color='#dc3545'
                ))
            
            fig.update_layout(
                title='Stock Movement Analysis',
                xaxis_title='Date',
                yaxis_title='Quantity',
                hovermode='x unified',
                height=500,
                barmode='relative'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View Stock History"):
                st.dataframe(product_stock, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 AI Inventory Insights")
            
            if metrics['days_on_hand'] < 7:
                st.markdown('<div class="alert-box">⚠️ <b>Low Stock Warning!</b><br>Current stock will last only ' + 
                           f"{metrics['days_on_hand']:.1f} days. Recommend placing urgent PO.</div>", unsafe_allow_html=True)
            elif metrics['days_on_hand'] > 30:
                st.markdown('<div class="alert-box">📦 <b>High Stock Level</b><br>Stock coverage is ' + 
                           f"{metrics['days_on_hand']:.1f} days. Consider promotion to reduce inventory.</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <b>Stock Level Optimal</b><br>Current stock level is healthy at ' + 
                           f"{metrics['days_on_hand']:.1f} days coverage.</div>", unsafe_allow_html=True)

# ==================== 🎯 PROMOTION PLANNING ====================
elif selected_module == "🎯 Promotion Planning":
    st.header("🎯 Promotion Planning")
    
    if scm.promotion_data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_promos = len(scm.promotion_data)
            st.metric("Total Promotions", total_promos)
        
        with col2:
            active = len(scm.promotion_data[scm.promotion_data['status'] == 'Active'])
            st.metric("Active", active)
        
        with col3:
            completed = len(scm.promotion_data[scm.promotion_data['status'] == 'Completed'])
            st.metric("Completed", completed)
        
        st.markdown("---")
        
        st.subheader("📋 Promotion List")
        
        promo_display = scm.promotion_data.copy()
        if 'target_sales' in promo_display.columns and 'actual_sales' in promo_display.columns:
            promo_display['ROI (%)'] = ((promo_display['actual_sales'] / promo_display['target_sales'] - 1) * 100).round(1)
        
        st.dataframe(promo_display, use_container_width=True, height=300)
        
        st.markdown("---")
        
        st.subheader("📊 Promotion Performance")
        
        completed_promos = scm.promotion_data[scm.promotion_data['status'] == 'Completed']
        
        if len(completed_promos) > 0 and 'target_sales' in completed_promos.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=completed_promos['promotion_name'],
                y=completed_promos['target_sales'],
                name='Target',
                marker_color='lightgray'
            ))
            
            fig.add_trace(go.Bar(
                x=completed_promos['promotion_name'],
                y=completed_promos['actual_sales'],
                name='Actual',
                marker_color='#667eea'
            ))
            
            fig.update_layout(
                title='Promotion Target vs Actual',
                xaxis_title='Promotion',
                yaxis_title='Sales',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== 📋 SUPPLY PLANNING ====================
elif selected_module == "📋 Supply Planning":
    st.header("📋 Supply Planning & Purchase Orders")
    
    if scm.po_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pos = len(scm.po_data)
            st.metric("Total POs", total_pos)
        
        with col2:
            pending = len(scm.po_data[scm.po_data['status'] == 'Pending'])
            st.metric("Pending", pending)
        
        with col3:
            in_transit = len(scm.po_data[scm.po_data['status'] == 'In Transit'])
            st.metric("In Transit", in_transit)
        
        with col4:
            received = len(scm.po_data[scm.po_data['status'] == 'Received'])
            st.metric("Received", received)
        
        st.markdown("---")
        
        st.subheader("📋 Purchase Order List")
        
        status_filter = st.multiselect(
            "Filter by Status",
            ['Pending', 'In Transit', 'Received'],
            default=['Pending', 'In Transit']
        )
        
        filtered_po = scm.po_data[scm.po_data['status'].isin(status_filter)]
        
        st.dataframe(filtered_po, use_container_width=True, height=400)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 PO by Product")
            po_by_product = scm.po_data.groupby('product_id')['order_qty'].sum().reset_index()
            
            fig = px.pie(po_by_product, values='order_qty', names='product_id',
                        title='Purchase Orders by Product')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Average Lead Time")
            received_pos = scm.po_data[scm.po_data['status'] == 'Received']
            
            if len(received_pos) > 0 and 'lead_time_days' in received_pos.columns:
                avg_lead_time = received_pos.groupby('product_id')['lead_time_days'].mean().reset_index()
                
                fig = px.bar(avg_lead_time, x='product_id', y='lead_time_days',
                            title='Average Lead Time by Product',
                            labels={'lead_time_days': 'Days', 'product_id': 'Product'},
                            color='lead_time_days',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

# ==================== 📊 REPORTS ====================
elif selected_module == "📊 Reports":
    st.header("📊 Reports & Analytics")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Sales Summary", "Inventory Report", "Promotion Performance", "Supply Chain Report"]
    )
    
    st.markdown("---")
    
    if report_type == "Sales Summary" and scm.sales_data is not None:
        st.subheader("📈 Sales Summary Report")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=scm.sales_data['date'].min())
        with col2:
            end_date = st.date_input("To Date", value=scm.sales_data['date'].max())
        
        filtered_sales = scm.sales_data[
            (scm.sales_data['date'] >= pd.to_datetime(start_date)) & 
            (scm.sales_data['date'] <= pd.to_datetime(end_date))
        ]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'total_amount' in filtered_sales.columns:
                st.metric("Total Revenue", f"฿{filtered_sales['total_amount'].sum():,.0f}")
        
        with col2:
            st.metric("Total Units Sold", f"{filtered_sales['sales_qty'].sum():,.0f}")
        
        with col3:
            if 'total_amount' in filtered_sales.columns:
                daily_avg = filtered_sales.groupby('date')['total_amount'].sum().mean()
                st.metric("Avg Daily Revenue", f"฿{daily_avg:,.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'total_amount' in filtered_sales.columns:
                sales_by_product = filtered_sales.groupby('product_id')['total_amount'].sum().reset_index()
                fig = px.bar(sales_by_product, x='product_id', y='total_amount',
                            title='Sales by Product',
                            labels={'total_amount': 'Revenue (฿)', 'product_id': 'Product'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'channel' in filtered_sales.columns and 'total_amount' in filtered_sales.columns:
                sales_by_channel = filtered_sales.groupby('channel')['total_amount'].sum().reset_index()
                fig = px.pie(sales_by_channel, values='total_amount', names='channel',
                            title='Sales by Channel')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        csv = filtered_sales.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Download Sales Report (CSV)",
            csv,
            f"sales_report_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    elif report_type == "Inventory Report" and scm.stock_data is not None:
        st.subheader("📦 Inventory Report")
        
        inventory_summary = []
        for product in scm.stock_data['product_id'].unique():
            metrics = scm.calculate_stock_metrics(product)
            if metrics:
                inventory_summary.append({
                    'Product': product,
                    'Current Stock': metrics['current_stock'],
                    'Days on Hand': metrics['days_on_hand'],
                    'Stock Turnover': metrics['stock_turnover'],
                    'Stock Value (฿)': metrics['stock_value'],
                    'Status': metrics['status']
                })
        
        inventory_df = pd.DataFrame(inventory_summary)
        
        st.dataframe(inventory_df, use_container_width=True)
        
        csv = inventory_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Download Inventory Report (CSV)",
            csv,
            f"inventory_report_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏭 <strong>Supply Chain Management System</strong> | Powered by AI & Advanced Analytics</p>
    <p>Integrated Solution for Sales Forecasting, Inventory, Promotions & Supply Planning</p>
</div>
""", unsafe_allow_html=True)
