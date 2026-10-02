"""
UI Components & Design System
Vibrant Healthcare Design System with Colorful Cards, Gradients, and Micro-interactions
100% English Language - Hospital & Decision-Support Grade
"""

import streamlit as st

def apply_custom_css():
    """Injects high-end medical software styling with vibrant gradients and modern micro-animations."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Main Container */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2563eb;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.03);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #334155;
        padding: 6px 10px;
        border-radius: 8px;
        transition: background 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #f1f5f9;
        color: #2563eb;
    }

    /* Top Navigation Header Bar */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 12px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.04);
    }

    /* Premium Health Cards */
    .health-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 4px 14px -3px rgba(0, 0, 0, 0.05);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }
    .health-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -4px rgba(37, 99, 235, 0.12);
        border-color: #cbd5e1;
    }

    /* Vital Metric Card with Dynamic Colored Top Accents */
    .vital-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 4px 14px -3px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .vital-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 30px -5px rgba(0, 0, 0, 0.08);
    }
    .vital-accent-rose {
        border-top: 4px solid #f43f5e;
    }
    .vital-accent-blue {
        border-top: 4px solid #3b82f6;
    }
    .vital-accent-teal {
        border-top: 4px solid #10b981;
    }

    .vital-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .vital-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.01em;
    }
    .vital-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .vital-unit {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        margin-left: 4px;
    }
    .vital-badge {
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-positive {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .badge-neutral {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
    }
    .badge-stable {
        background: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
    }
    .badge-warning {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
    }
    .badge-danger {
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }

    /* Vibrant Mesh Gradient Banner for Predictor */
    .predictor-banner {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 45%, #9333ea 100%);
        border-radius: 20px;
        padding: 38px 28px;
        text-align: center;
        color: #ffffff;
        margin-bottom: 26px;
        box-shadow: 0 14px 35px -6px rgba(99, 102, 241, 0.4);
        position: relative;
        overflow: hidden;
    }
    .predictor-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 60%);
        pointer-events: none;
    }
    .predictor-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .predictor-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        margin-top: 8px;
        font-weight: 500;
    }

    /* Colorful Insights Cards */
    .insights-box-warning {
        background: linear-gradient(135deg, #fef08a 0%, #fde047 100%);
        border: 1px solid #eab308;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: #713f12;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(234, 179, 8, 0.2);
        margin-bottom: 16px;
    }
    .insights-box-healthy {
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%);
        border: 1px solid #22c55e;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: #14532d;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
        margin-bottom: 16px;
    }
    .insights-box-danger {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: #7f1d1d;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
        margin-bottom: 16px;
    }

    /* Health Tips Box */
    .health-tips-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .disclaimer-card {
        background: #fffbeb;
        border: 1.5px solid #fde68a;
        border-radius: 14px;
        padding: 16px;
        font-size: 0.85rem;
        color: #92400e;
        line-height: 1.55;
    }

    /* History Dark Container (Screenshot 5) with Neon Accent */
    .history-dark-card {
        background: radial-gradient(circle at 15% 25%, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 28px;
        color: #f8fafc;
        margin-bottom: 24px;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.4);
    }
    .history-lock-badge {
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        padding: 8px 16px;
        border-radius: 12px;
        display: inline-block;
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 20px;
    }

    /* Primary Gradient Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.8rem;
        font-weight: 700;
        font-size: 0.98rem;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
        transition: all 0.25s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.5);
        transform: translateY(-2px);
    }

    /* Form Input Fields */
    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        background: #ffffff !important;
    }
    div[data-testid="stNumberInput"] input:focus, div[data-testid="stTextInput"] input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }

    /* Metric Boxes in Dashboard */
    .kpi-stat-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px -2px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-stat-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 4px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_top_bar(current_date="October 24, 2026"):
    """Renders the HealthSync top search and action bar matching Image 1."""
    st.markdown(f"""
    <div class="top-header-bar">
        <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
            <span style="color: #6366f1; font-size: 1.15rem;">🔍</span>
            <input type="text" placeholder="Search patients, medical records, or physiological biomarkers..." 
                style="border: none; outline: none; background: transparent; width: 85%; font-size: 0.92rem; color: #334155; font-family: inherit;" />
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="background: #f1f5f9; padding: 8px 11px; border-radius: 50%; color: #64748b; font-size: 0.95rem; cursor: pointer; transition: all 0.2s ease;">🔔</span>
            <span style="background: #f1f5f9; padding: 8px 11px; border-radius: 50%; color: #64748b; font-size: 0.95rem; cursor: pointer; transition: all 0.2s ease;">❓</span>
            <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); color: #1d4ed8; font-weight: 700; font-size: 0.85rem; padding: 7px 16px; border-radius: 10px; border: 1px solid #bfdbfe; box-shadow: 0 2px 6px rgba(37,99,235,0.08);">
                📅 {current_date}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_doctor_greeting(doctor_name="Dr. Williams", appointment_count=12, urgent_count=4):
    """Renders the welcome heading matching Image 1."""
    st.markdown(f"""
    <div style="margin-bottom: 22px;">
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 800; color: #0f172a; margin: 0 0 6px 0; letter-spacing: -0.5px;">
            Welcome back, {doctor_name}
        </h1>
        <p style="font-size: 0.98rem; color: #64748b; margin: 0; font-weight: 500;">
            You have <b style="color: #2563eb; font-weight: 700;">{appointment_count} patient screenings</b> and <b style="color: #ef4444; font-weight: 700;">{urgent_count} high-risk reviews</b> scheduled for today.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_vital_metric_card(title, value, unit, icon, badge_text, badge_class="badge-positive", accent_class="vital-accent-rose", sparkline_svg=None):
    """Renders a vital metric card with sparkline SVG matching Image 1."""
    sparkline_html = sparkline_svg if sparkline_svg else ""
    return f"""
    <div class="vital-card {accent_class}">
        <div class="vital-header">
            <span style="font-size: 1.35rem;">{icon}</span>
            <span class="vital-badge {badge_class}">{badge_text}</span>
        </div>
        <div class="vital-title">{title}</div>
        <div style="margin-top: 4px; margin-bottom: 12px;">
            <span class="vital-value">{value}</span>
            <span class="vital-unit">{unit}</span>
        </div>
        <div style="height: 40px; width: 100%;">
            {sparkline_html}
        </div>
    </div>
    """
