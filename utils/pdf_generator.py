"""
Clinical PDF Report Generator for AI-Based Diabetes Prediction System
Generates a PDF screening report with patient metrics, model risk score,
clinical biomarkers, recommendations, and a clear educational disclaimer.
This report is for educational/screening purposes only - not a medical diagnosis.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(patient_data, prediction_result):
    """
    Generates a PDF bytes buffer containing the clinical report.
    patient_data dict keys:
        name, age, gender, pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, pedigree
    prediction_result dict keys:
        prediction (0 or 1), probability (0.0 to 1.0),
        risk_level ('Low Risk', 'Moderate Risk', 'High Risk'),
        model_name, notes, record_id
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom Color Palette
    primary_color = colors.HexColor("#1E3A8A")    # Deep Navy Blue
    secondary_color = colors.HexColor("#0284C7")  # Medical Sky Blue
    accent_green = colors.HexColor("#16A34A")     # Healthy Green
    accent_red = colors.HexColor("#DC2626")       # High Risk Red
    accent_amber = colors.HexColor("#D97706")     # Moderate Amber
    bg_light = colors.HexColor("#F8FAFC")
    text_dark = colors.HexColor("#0F172A")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        alignment=0
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_dark
    )

    bold_style = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=text_dark
    )

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>AI-BASED DIABETES PREDICTION SYSTEM</b><br/><font size=8 color='#64748B'>Machine Learning Screening Report — For Educational Use Only</font>", title_style),
            Paragraph(f"<b>Report ID:</b> DIA-{prediction_result.get('record_id', '999'):04d}<br/><b>Date:</b> {datetime.now().strftime('%d %b %Y, %I:%M %p')}<br/><b>Status:</b> Completed", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.0 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceBefore=0, spaceAfter=12))

    # 2. Patient Demographics Card
    patient_info = [
        [
            Paragraph("<b>Patient Name:</b>", bold_style), Paragraph(str(patient_data.get('name', 'N/A')), body_style),
            Paragraph("<b>Gender:</b>", bold_style), Paragraph(str(patient_data.get('gender', 'N/A')), body_style)
        ],
        [
            Paragraph("<b>Age:</b>", bold_style), Paragraph(f"{patient_data.get('age', 'N/A')} Years", body_style),
            Paragraph("<b>Pregnancies:</b>", bold_style), Paragraph(str(patient_data.get('pregnancies', 0)), body_style)
        ],
        [
            Paragraph("<b>Evaluation Engine:</b>", bold_style), Paragraph(str(prediction_result.get('model_name', 'Random Forest')), body_style),
            Paragraph("<b>Assessment Type:</b>", bold_style), Paragraph("Predictive Multi-Feature ML", body_style)
        ]
    ]
    p_table = Table(patient_info, colWidths=[1.4 * inch, 2.2 * inch, 1.4 * inch, 2.2 * inch])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 12))

    # 3. AI Prediction Result Highlight Card
    pred_val = prediction_result.get('prediction', 0)
    risk_level = prediction_result.get('risk_level', 'Low Risk')
    prob_pct = round(prediction_result.get('probability', 0.0) * 100, 1)

    if pred_val == 1:
        res_text = "POSITIVE FOR HIGH DIABETES RISK"
        banner_bg = colors.HexColor("#FEE2E2")
        badge_color = accent_red
        rec_summary = "Please consult a licensed physician for formal clinical evaluation (e.g., Fasting Plasma Glucose, HbA1c tests)."
    elif risk_level == "Moderate Risk":
        res_text = "BORDERLINE / MODERATE DIABETES RISK DETECTED"
        banner_bg = colors.HexColor("#FEF3C7")
        badge_color = accent_amber
        rec_summary = "Lifestyle modifications and routine follow-up diagnostic testing are recommended."
    else:
        res_text = "LOW DIABETES RISK DETECTED"
        banner_bg = colors.HexColor("#DCFCE7")
        badge_color = accent_green
        rec_summary = "Model indicators appear within low-risk range. Maintain a balanced lifestyle and periodic health check-ups."

    # Model confidence: probability of the predicted class (max 99.5% — never claim 100%)
    conf_pct = round(min(max(prob_pct, round(100.0 - prob_pct, 1)), 99.5), 1)

    pred_card = [
        [
            Paragraph(f"<font color='{badge_color.hexval()}'><b>MODEL PREDICTION RESULT: {res_text}</b></font>", ParagraphStyle('ResHead', fontName='Helvetica-Bold', fontSize=12, leading=16)),
            Paragraph(f"<b>Estimated AI Risk Score:</b> <font color='{badge_color.hexval()}'><b>{prob_pct}%</b></font><br/><b>Stratification:</b> {risk_level}", ParagraphStyle('ResRisk', fontName='Helvetica', fontSize=10, leading=14, alignment=2))
        ],
        [
            Paragraph(f"<b>Screening Synopsis:</b> {rec_summary}", ParagraphStyle('Syn', fontName='Helvetica', fontSize=8.5, leading=12, textColor=text_dark)),
            Paragraph(f"<i>Model Confidence: {conf_pct}%</i>", ParagraphStyle('Syn2', fontName='Helvetica-Oblique', fontSize=8, leading=12, alignment=2, textColor=colors.HexColor("#64748B")))
        ]
    ]
    res_table = Table(pred_card, colWidths=[5.0 * inch, 2.2 * inch])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.0, badge_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 14))

    # 4. Clinical Health Parameters Breakdown Table
    story.append(Paragraph("<b>Recorded Clinical Biomarkers & Reference Standards</b>", h2_style))
    
    # Biomarkers evaluation
    glu = float(patient_data.get('glucose', 100))
    glu_stat = "Elevated" if glu >= 140 else ("Impaired" if glu >= 100 else "Normal")
    glu_col = accent_red if glu >= 140 else (accent_amber if glu >= 100 else accent_green)

    bp = float(patient_data.get('blood_pressure', 80))
    bp_stat = "High" if bp >= 90 else ("Pre-hypertension" if bp >= 80 else "Normal")
    bp_col = accent_red if bp >= 90 else (accent_amber if bp >= 80 else accent_green)

    bmi = float(patient_data.get('bmi', 24.0))
    bmi_stat = "Obese" if bmi >= 30 else ("Overweight" if bmi >= 25 else "Normal")
    bmi_col = accent_red if bmi >= 30 else (accent_amber if bmi >= 25 else accent_green)

    ins = float(patient_data.get('insulin', 80))
    ins_stat = "High" if ins > 166 else ("Normal" if ins >= 16 else "Low")
    ins_col = accent_amber if (ins > 166 or ins < 16) else accent_green

    skin = float(patient_data.get('skin_thickness', 20))
    ped = float(patient_data.get('pedigree', 0.5))
    age = int(patient_data.get('age', 30))

    param_rows = [
        [
            Paragraph("<b>Biomarker Parameter</b>", bold_style),
            Paragraph("<b>Patient Value</b>", bold_style),
            Paragraph("<b>Clinical Normal Range</b>", bold_style),
            Paragraph("<b>Status / Evaluation</b>", bold_style)
        ],
        [
            Paragraph("Plasma Glucose Concentration", body_style),
            Paragraph(f"<b>{glu:.1f} mg/dL</b>", body_style),
            Paragraph("70 - 99 (Fasting) / < 140 (Random)", body_style),
            Paragraph(f"<font color='{glu_col.hexval()}'><b>{glu_stat}</b></font>", body_style)
        ],
        [
            Paragraph("Diastolic Blood Pressure", body_style),
            Paragraph(f"<b>{bp:.1f} mm Hg</b>", body_style),
            Paragraph("60 - 80 mm Hg", body_style),
            Paragraph(f"<font color='{bp_col.hexval()}'><b>{bp_stat}</b></font>", body_style)
        ],
        [
            Paragraph("Body Mass Index (BMI)", body_style),
            Paragraph(f"<b>{bmi:.1f} kg/m²</b>", body_style),
            Paragraph("18.5 - 24.9 kg/m²", body_style),
            Paragraph(f"<font color='{bmi_col.hexval()}'><b>{bmi_stat}</b></font>", body_style)
        ],
        [
            Paragraph("2-Hour Serum Insulin", body_style),
            Paragraph(f"<b>{ins:.1f} μU/mL</b>", body_style),
            Paragraph("16 - 166 μU/mL", body_style),
            Paragraph(f"<font color='{ins_col.hexval()}'><b>{ins_stat}</b></font>", body_style)
        ],
        [
            Paragraph("Triceps Skinfold Thickness", body_style),
            Paragraph(f"<b>{skin:.1f} mm</b>", body_style),
            Paragraph("10 - 30 mm", body_style),
            Paragraph("<font color='#16A34A'><b>Evaluated</b></font>", body_style)
        ],
        [
            Paragraph("Diabetes Pedigree Function", body_style),
            Paragraph(f"<b>{ped:.3f}</b>", body_style),
            Paragraph("0.08 - 0.50 (Low Genetic Risk)", body_style),
            Paragraph("<font color='#D97706'><b>Genetic Factor</b></font>" if ped > 0.5 else "<font color='#16A34A'><b>Average</b></font>", body_style)
        ],
        [
            Paragraph("Patient Age", body_style),
            Paragraph(f"<b>{age} years</b>", body_style),
            Paragraph("N/A", body_style),
            Paragraph("Adult", body_style)
        ]
    ]

    p_param_table = Table(param_rows, colWidths=[2.3 * inch, 1.4 * inch, 2.2 * inch, 1.3 * inch])
    p_param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(p_param_table)
    story.append(Spacer(1, 14))

    # 5. Personalized Recommendations
    story.append(Paragraph("<b>Personalized Clinical Recommendations & Preventative Guide</b>", h2_style))
    recs = []
    if pred_val == 1 or risk_level == "High Risk":
        recs = [
            "• <b>Clinical Verification:</b> Schedule an official Fasting Plasma Glucose (FPG) test and Glycated Hemoglobin (HbA1c) test with a physician.",
            "• <b>Dietary Management:</b> Minimize refined carbohydrates, sweets, sweetened beverages, and saturated fats. Adopt a low-glycemic index diet with high dietary fiber.",
            "• <b>Physical Activity:</b> Aim for at least 150 minutes of moderate-intensity aerobic exercise per week (e.g. brisk walking, cycling, swimming).",
            "• <b>Biomarker Tracking:</b> Maintain a daily or weekly log of capillary blood glucose and resting blood pressure."
        ]
    elif risk_level == "Moderate Risk":
        recs = [
            "• <b>Preventative Screening:</b> Undergo annual diabetes risk checkups including lipid profile and oral glucose tolerance testing.",
            "• <b>Weight Optimization:</b> Target a progressive 5% to 7% body weight reduction if BMI is above 25 kg/m².",
            "• <b>Nutritional Balance:</b> Increase leafy greens, whole grains, nuts, and lean proteins; avoid ultra-processed foods.",
            "• <b>Hydration & Rest:</b> Ensure 7-8 hours of quality sleep nightly to optimize insulin sensitivity."
        ]
    else:
        recs = [
            "• <b>Health Preservation:</b> Continue your current positive physical routine and nutritious dietary practices.",
            "• <b>Periodic Monitoring:</b> Re-evaluate your biomarkers every 12 to 24 months, particularly if family medical history changes.",
            "• <b>Active Lifestyle:</b> Avoid sedentary habits; take short walking breaks during prolonged desk work."
        ]

    for r in recs:
        story.append(Paragraph(r, body_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))

    # 6. Legal Disclaimer & Sign-off Block
    disclaimer_text = (
        "<b>LEGAL & EDUCATIONAL DISCLAIMER:</b> This report is generated by an Artificial Intelligence and Machine Learning "
        "screening algorithm trained on the Pima Indians Diabetes Dataset. It is provided strictly for educational "
        "and preliminary screening purposes ONLY. It does NOT constitute a medical diagnosis, clinical prescription, "
        "treatment plan, or physician consultation. This tool cannot replace professional medical evaluation. "
        "A licensed healthcare practitioner must be consulted for any formal clinical diagnosis or medical decision-making."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', fontName='Helvetica-Oblique', fontSize=7.5, leading=10, textColor=colors.HexColor("#64748B"))))
    story.append(Spacer(1, 14))

    # Doctor Sign-off block
    sign_data = [
        [
            Paragraph("<b>Automated Verification:</b><br/>Algorithm: Scikit-Learn Pipeline<br/>Validation: 5-Fold Stratified CV", ParagraphStyle('Sign1', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor("#475569"))),
            Paragraph("<b>Consulting Physician / Reviewer:</b><br/><br/>________________________________________<br/>Signature & Medical Registration Stamp", ParagraphStyle('Sign2', fontName='Helvetica', fontSize=8, leading=11, alignment=2, textColor=colors.HexColor("#475569")))
        ]
    ]
    sign_table = Table(sign_data, colWidths=[3.6 * inch, 3.6 * inch])
    sign_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(sign_table)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
