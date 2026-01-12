#!/usr/bin/env python3
"""Generate PDF report tables for CAP+OCI v6 results."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

def create_pdf_report(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=15*mm,
        bottomMargin=15*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=TA_LEFT,
        spaceAfter=8,
        spaceBefore=16
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_LEFT
    )

    elements = []

    # Title
    elements.append(Paragraph("CAP+OCI v0.3.6p15.2ABv6 - Full Evaluation Results", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 10*mm))

    # Summary Table
    elements.append(Paragraph("Summary", subtitle_style))

    summary_data = [
        ['Environment', 'CLAIM_A\n(Onset)', 'CLAIM_B\n(Robust)', 'CLAIM_B+\n(Strong)', 'CLAIM_C\n(Mechanistic)', 'crossing', 'max_onset', 'min_inadeq'],
        ['EnvA_grid', 'True', 'True', 'True', 'True', 'True (down)', '1.000', '0.000'],
        ['EnvB_continuous', 'True', 'True', 'True', 'True', 'True (up)', '0.670', '0.000'],
    ]

    summary_table = Table(summary_data, colWidths=[45*mm, 22*mm, 22*mm, 22*mm, 28*mm, 28*mm, 22*mm, 22*mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D6DCE5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#D6DCE5'), colors.HexColor('#E9ECF1')]),
    ]))
    elements.append(summary_table)

    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph("<b>claim_ready = True</b> (maps to CLAIM_A ∧ CLAIM_B ∧ CLAIM_C in both envs)", normal_style))
    elements.append(Paragraph("<b>claim_b_strong = True</b> (CLAIM_B+ in both envs, report-only)", normal_style))
    elements.append(Spacer(1, 3*mm))

    # Note style for annotations
    note_style = ParagraphStyle(
        'NoteStyle',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#555555'),
        leftIndent=10
    )
    elements.append(Paragraph("† CLAIM_B+ (Strong) = True means at least one γ-regime achieves AND condition (meta>0 AND rec>0) per environment, not all γ.", note_style))
    elements.append(Paragraph("† crossing is evaluated on CLAIM_B (weak/OR); CLAIM_B+ is a report-only label and does not affect crossing or claim_ready.", note_style))
    elements.append(Spacer(1, 8*mm))

    # EnvA_grid Table
    elements.append(Paragraph("EnvA_grid - Per-γ Results", subtitle_style))

    envA_data = [
        ['γ', 'onset_rate', 'n_onset', 'OR_rate', 'OR_LCB', 'AND_rate', 'AND_LCB', 'n_meta', 'n_rec', 'n_and', 'robust(OR)', 'strong(AND)'],
        ['1.00', '0.000', '0', '0.000', '0.000', '0.000', '0.000', '0', '0', '0', 'False', 'False'],
        ['0.75', '0.000', '0', '0.000', '0.000', '0.000', '0.000', '0', '0', '0', 'False', 'False'],
        ['0.50', '0.348', '32', '0.906', '0.819', '0.469', '0.360', '29', '15', '15', 'True', 'True'],
        ['0.25', '0.962', '76', '0.895', '0.841', '0.263', '0.204', '68', '25', '20', 'True', 'True'],
        ['0.00', '1.000', '73', '0.781', '0.713', '0.247', '0.188', '57', '21', '18', 'True', 'True'],
    ]

    envA_table = Table(envA_data, colWidths=[12*mm, 20*mm, 16*mm, 18*mm, 18*mm, 20*mm, 20*mm, 16*mm, 14*mm, 14*mm, 22*mm, 24*mm])
    envA_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#D6DCE5'), colors.HexColor('#E9ECF1')]),
        # Highlight PASS rows
        ('BACKGROUND', (10, 3), (10, 5), colors.HexColor('#C6EFCE')),
        ('BACKGROUND', (11, 3), (11, 5), colors.HexColor('#C6EFCE')),
        # Highlight FAIL rows
        ('BACKGROUND', (10, 1), (10, 2), colors.HexColor('#FFC7CE')),
        ('BACKGROUND', (11, 1), (11, 2), colors.HexColor('#FFC7CE')),
    ]))
    elements.append(envA_table)
    elements.append(Paragraph("Crossing(down): PASS ≤ 0.5, FAIL ≥ 0.75", normal_style))
    elements.append(Paragraph("† Rows with n_onset=0: rate/LCB values are undefined; displayed as 0.000 for convenience (do not contribute to PASS).", note_style))
    elements.append(Spacer(1, 6*mm))

    # EnvB_continuous Table
    elements.append(Paragraph("EnvB_continuous - Per-γ Results", subtitle_style))

    envB_data = [
        ['γ', 'onset_rate', 'n_onset', 'OR_rate', 'OR_LCB', 'AND_rate', 'AND_LCB', 'n_meta', 'n_rec', 'n_and', 'robust(OR)', 'strong(AND)'],
        ['1.00', '0.430', '39', '0.872', '0.790', '0.615', '0.510', '34', '28', '24', 'True', 'True'],
        ['0.75', '0.640', '61', '0.639', '0.560', '0.393', '0.320', '39', '28', '24', 'True', 'True'],
        ['0.50', '0.670', '67', '0.388', '0.320', '0.134', '0.090', '26', '20', '9', 'True', 'False'],
        ['0.25', '0.580', '58', '0.224', '0.160', '0.034', '0.010', '13', '10', '2', 'True', 'False'],
        ['0.00', '0.010', '1', '0.000', '0.000', '0.000', '0.000', '0', '0', '0', 'False', 'False'],
    ]

    envB_table = Table(envB_data, colWidths=[12*mm, 20*mm, 16*mm, 18*mm, 18*mm, 20*mm, 20*mm, 16*mm, 14*mm, 14*mm, 22*mm, 24*mm])
    envB_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#D6DCE5'), colors.HexColor('#E9ECF1')]),
        # Highlight robust(OR) PASS
        ('BACKGROUND', (10, 1), (10, 4), colors.HexColor('#C6EFCE')),
        ('BACKGROUND', (10, 5), (10, 5), colors.HexColor('#FFC7CE')),
        # Highlight strong(AND)
        ('BACKGROUND', (11, 1), (11, 2), colors.HexColor('#C6EFCE')),
        ('BACKGROUND', (11, 3), (11, 5), colors.HexColor('#FFC7CE')),
    ]))
    elements.append(envB_table)
    elements.append(Paragraph("Crossing(up): FAIL ≤ 0.0, PASS ≥ 0.25", normal_style))
    elements.append(Paragraph("† Rows with n_onset ≤ min_required (20 for full_eval): rate/LCB may be unreliable; shown for transparency.", note_style))

    # Page break for sensitivity analysis
    elements.append(PageBreak())

    # Sensitivity Analysis Table
    elements.append(Paragraph("Threshold Sensitivity Analysis", subtitle_style))
    elements.append(Paragraph("All 32 parameter combinations resulted in claim_ready = True", normal_style))
    elements.append(Spacer(1, 5*mm))

    sens_header = ['R_PASS', 'R_STRONG', 'θ_lead', 'θ_rec', 'claim_ready', 'claim_b_strong']
    sens_data = [sens_header]

    for r_pass in ['0.10', '0.15', '0.20', '0.25']:
        for r_strong in ['0.05', '0.10']:
            for theta_lead in ['0.00', '-0.05']:
                for theta_rec in ['0.00', '-0.05']:
                    sens_data.append([r_pass, r_strong, theta_lead, theta_rec, 'True', 'True'])

    # Show only representative subset (8 rows)
    sens_subset = [sens_header] + sens_data[1:5] + ['...'] + sens_data[-4:]
    sens_subset_clean = []
    for row in sens_subset:
        if row == '...':
            sens_subset_clean.append(['...', '...', '...', '...', '...', '...'])
        else:
            sens_subset_clean.append(row)

    sens_table = Table(sens_subset_clean, colWidths=[25*mm, 25*mm, 20*mm, 20*mm, 30*mm, 30*mm])
    sens_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#D6DCE5'), colors.HexColor('#E9ECF1')]),
        ('BACKGROUND', (4, 1), (5, -1), colors.HexColor('#C6EFCE')),
    ]))
    elements.append(sens_table)
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph("Note: All 32 combinations (R_PASS: 0.10-0.25, R_STRONG: 0.05-0.10, θ: 0.00 to -0.05) passed.", normal_style))

    elements.append(Spacer(1, 10*mm))

    # Conclusion box
    elements.append(Paragraph("Conclusions", subtitle_style))
    conclusion_data = [
        ['Metric', 'Status', 'Description'],
        ['claim_ready', 'PASS', 'Both environments achieve CLAIM_A/B/C with crossing'],
        ['claim_b_strong', 'PASS', 'Both environments achieve AND condition (meta>0 AND rec>0)'],
        ['Threshold Robustness', 'PASS', 'Results stable across R_PASS 0.10-0.25 (2.5x range)'],
        ['Reproducibility', 'PASS', '3 independent runs all achieved claim_ready=True'],
    ]

    concl_table = Table(conclusion_data, colWidths=[45*mm, 20*mm, 150*mm])
    concl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'CENTER'),
        ('ALIGN', (2, 0), (2, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#D6DCE5'), colors.HexColor('#E9ECF1')]),
        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#C6EFCE')),
    ]))
    elements.append(concl_table)

    # Build PDF
    doc.build(elements)
    print(f"PDF generated: {output_path}")

if __name__ == "__main__":
    output_path = r"C:\Users\key\Desktop\unko\CAP_OCI_v6_Results.pdf"
    create_pdf_report(output_path)
