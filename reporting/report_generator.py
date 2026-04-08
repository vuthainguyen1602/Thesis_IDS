#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import base64
import pandas as pd

def export_results_to_html(
    results: dict, title: str = "Experiment Results",
    output_path: str = "report.html", chart_paths: list = None,
) -> None:
    import base64
    
    df = pd.DataFrame(results).T
    display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                    "training_time", "prediction_time", "model_size_mb"]
    available = [c for c in display_cols if c in df.columns]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f7f9; color: #333; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
            h2 {{ color: #5f6368; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; color: #202124; font-weight: 600; }}
            tr:hover {{ background-color: #f1f3f4; }}
            .metric {{ font-weight: bold; color: #1a73e8; }}
            .chart-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 30px; }}
            .chart-box {{ flex: 1; min-width: 450px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }}
            .chart-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #70757a; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        {" ".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in available)}
                    </tr>
                </thead>
                <tbody>
    """
    
    for name, row in df.iterrows():
        html += f"<tr><td><strong>{name}</strong></td>"
        for col_name in available:
            val = row.get(col_name)
            if pd.isna(val) or val is None:
                formatted = "N/A"
            elif col_name in ["training_time", "prediction_time"]:
                formatted = f"{val:.3f}s"
            elif col_name == "model_size_mb":
                formatted = f"{val:.3f} MB"
            else:
                formatted = f"{val:.6f}"
            html += f"<td>{formatted}</td>"
        html += "</tr>"
        
    html += """
                </tbody>
            </table>
    """
    
    if chart_paths:
        html += '<h2>Visualizations</h2><div class="chart-container">'
        for cp in chart_paths:
            if os.path.exists(cp):
                try:
                    with open(cp, "rb") as f:
                        data = base64.b64encode(f.read()).decode()
                        filename = os.path.basename(cp)
                        html += f"""
                        <div class="chart-box">
                            <h3>{filename}</h3>
                            <img src="data:image/png;base64,{data}" alt="{filename}">
                        </div>
                        """
                except Exception as e:
                    html += f"<p>Error embedding {cp}: {str(e)}</p>"
        html += '</div>'
        
    html += f"""
            <div class="footer">
                <p>IDS Thesis Binary Prediction - Apache Spark Result Report</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[INFO] Report exported to: {output_path}")


def export_multi_section_report(
    sections: list, title: str = "Experiment Results",
    output_path: str = "report.html",
) -> None:
    import base64
    
    css = """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f7f9; color: #333; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; text-align: center; margin-bottom: 40px; }
        .section { margin-bottom: 60px; padding-top: 20px; border-top: 1px solid #eee; }
        h2 { color: #1a73e8; margin-top: 0; background: #e8f0fe; padding: 10px 20px; border-radius: 4px; }
        h3 { color: #5f6368; margin-top: 25px; border-left: 4px solid #1a73e8; padding-left: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; color: #202124; font-weight: 600; }
        tr:hover { background-color: #f1f3f4; }
        .chart-container { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 25px; }
        .chart-box { flex: 1; min-width: 450px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }
        .chart-box img { max-width: 100%; height: auto; border-radius: 4px; }
        .footer { margin-top: 50px; font-size: 12px; color: #70757a; text-align: center; }
    """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>{css}</style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p style="text-align:center;">Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    """

    for sec in sections:
        sec_title = sec.get("section_title", "Section")
        results = sec.get("results", {})
        chart_paths = sec.get("chart_paths", [])
        
        df = pd.DataFrame(results).T
        display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                        "training_time", "prediction_time", "model_size_mb"]
        available = [c for c in display_cols if c in df.columns]

        html += f"""
            <div class="section">
                <h2>{sec_title}</h2>
                <h3>Summary Results</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            {" ".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in available)}
                        </tr>
                    </thead>
                    <tbody>
        """

        for name, row in df.iterrows():
            html += f"<tr><td><strong>{name}</strong></td>"
            for col_name in available:
                val = row.get(col_name)
                if pd.isna(val) or val is None:
                    formatted = "N/A"
                elif col_name in ["training_time", "prediction_time"]:
                    formatted = f"{val:.3f}s"
                elif col_name == "model_size_mb":
                    formatted = f"{val:.3f} MB"
                else:
                    formatted = f"{val:.6f}"
                html += f"<td>{formatted}</td>"
            html += "</tr>"

        html += "</tbody></table>"

        if chart_paths:
            html += '<div class="chart-container">'
            for cp in chart_paths:
                if os.path.exists(cp):
                    try:
                        with open(cp, "rb") as f:
                            data = base64.b64encode(f.read()).decode()
                            filename = os.path.basename(cp)
                            html += f"""
                            <div class="chart-box">
                                <p style="font-weight:600;margin:5px 0;">{filename}</p>
                                <img src="data:image/png;base64,{data}" alt="{filename}">
                            </div>
                            """
                    except Exception as e:
                        html += f"<p>Error embedding {cp}: {str(e)}</p>"
            html += '</div>'
        
        html += "</div>"

    html += f"""
            <div class="footer">
                <p>IDS Thesis Binary Prediction - Comprehensive Experiment Report</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[INFO] Comprehensive Report exported to: {output_path}")

