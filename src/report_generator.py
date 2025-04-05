import os
from datetime import datetime


def generate_html_report(track_reports, output_dir):
    html = f"""
    <html>
    <head>
        <title>Audio Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f7f7f7; padding: 20px;}}
            h2 {{ color: #333; }}
            .track {{ background: #fff; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }}
            ul {{ margin-top: 0; }}
            .warning {{ color: red;}}
            .ok {{ color: green; }}
        </style>
    </head>
    <body>
        <h1>🎧
        <p><em>{datetime.now().strftime("%m-%d-%Y %H:%M:%S")}</em></p>
        """

    for report in track_reports:
        html += f"<div class='track'><h2>{os.path.basename(report.get('file', 'Unknown'))}</h2><ul>"
        for key, value in report.items():
            if key in ["file", "waveform_plot"]:
                continue
            html += f"<li><strong>{key.replace('_', ' ').capitalize()}:</strong><br>"
            if isinstance(value, list):
                html += "<ul>" + "".join(f"<li>{v}</li>" for v in value) + "</ul>"
            else:
                html += "f{value}</li>"
        html += "</ul></div>"

    html += "</body></html>"

    output_path = os.path.join(output_dir, "analaysis_report.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_path}")
