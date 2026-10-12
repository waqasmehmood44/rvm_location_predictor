"""
Helper script to append new manual test results to factors_accuracy_test_file.csv and Excel workbook.
"""

import os
import pandas as pd

def append_manual_test(summary_dict, factor_list, monthly_list=None):
    csv_summary_path = "reports/factors_accuracy_test_file.csv"
    csv_detailed_path = "reports/factors_accuracy_test_file_detailed.csv"
    excel_path = "reports/factors_accuracy_test_file.xlsx"

    # Load existing or initialize
    if os.path.exists(csv_summary_path):
        df_summary = pd.read_csv(csv_summary_path)
        next_id = len(df_summary) + 1
    else:
        df_summary = pd.DataFrame()
        next_id = 1

    if os.path.exists(csv_detailed_path):
        df_factors = pd.read_csv(csv_detailed_path)
    else:
        df_factors = pd.DataFrame()

    summary_dict["Test_ID"] = next_id
    for f in factor_list:
        f["Test_ID"] = next_id

    df_summary = pd.concat([df_summary, pd.DataFrame([summary_dict])], ignore_index=True)
    df_factors = pd.concat([df_factors, pd.DataFrame(factor_list)], ignore_index=True)

    df_summary.to_csv(csv_summary_path, index=False, encoding="utf-8-sig")
    df_factors.to_csv(csv_detailed_path, index=False, encoding="utf-8-sig")

    # Update Excel
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Manual_Tests_Summary", index=False)
            df_factors.to_excel(writer, sheet_name="Factors_Accuracy_Clusters", index=False)
    except PermissionError:
        alt_path = "reports/factors_accuracy_test_file_latest.xlsx"
        with pd.ExcelWriter(alt_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Manual_Tests_Summary", index=False)
            df_factors.to_excel(writer, sheet_name="Factors_Accuracy_Clusters", index=False)
        print(f"Note: {excel_path} was locked (open in Excel). Saved to {alt_path} instead.")

    print(f"Appended Test #{next_id} ({summary_dict.get('Store_Name')}) successfully.")

if __name__ == "__main__":
    print("Helper script ready for subsequent manual tests.")
