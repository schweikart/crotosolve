# Set main tex file
@default_files = ('thesis.tex');

# Compile to pdf directly
$pdf_mode = 1;

# Add KITreport classes
ensure_path('TEXINPUTS', './sdqthesis//');
