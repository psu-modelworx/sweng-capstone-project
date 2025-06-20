from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime

class ReportingEngine:
    """Generates a PDF report for preprocessing info and model generation info."""
    def __init__(self, preprocessor, modeler):
        """ initializes the reporting engine with preprocessor and modeler objects."""
        self.preprocessor = preprocessor
        self.modeler = modeler
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_font("Helvetica", size=12)
    # Utility Methods
    def section_header(self, text):
        self.pdf.set_font("Helvetica", 'B', 14)
        self.pdf.multi_cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font("Helvetica", size=12)
        self.pdf.ln(2)

    def subsection(self, text):
        self.pdf.set_font("Helvetica", 'B', 12)
        self.pdf.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font("Helvetica", size=12)

    def add_bullet(self, text):
        self.pdf.multi_cell(10)
        self.pdf.multi_cell(0, 8, f"- {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
