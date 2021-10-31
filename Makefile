debug:
	python3 main.py DEBUG

run:
	python3 main.py INFO

app:
	python3 app.py


generate_pdf:
	markdown-pdf README.md
	mv README.pdf report.pdf

