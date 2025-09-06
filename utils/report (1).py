from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def build_credit_report(user, summary: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "LoanIQ Credit Report")
    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"User: {user['username']}  |  Role: {user['role']}")
    y -= 18
    c.drawString(40, y, f"Summary Date: {summary.get('date','')}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Credit Summary:")
    y -= 16
    c.setFont("Helvetica", 10)
    
    # Add summary data
    for key, value in summary.items():
        if key != 'date':
            c.drawString(60, y, f"{key}: {value}")
            y -= 14
            if y < 100:  # Simple page break
                c.showPage()
                y = h - 40

    c.save()
    return buf.getvalue()