"""
Synchronizes the latest benchmark stores:
- Cantù (PF00001305) with updated geocoded underwritten baseline (978.8 b/d)
- Portoferraio (PF00001460) (1,018.3 b/d)
- Prato - Parco Prato (PF00001636) (4,679.5 b/d)
Updates benchmark_tests.json, benchmark_tests.csv, dashboard_monthly_records.csv,
dashboard_dow_records.csv, reports/accuracy_dashboard.html, and model_accuracy_dashboard.png.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from scripts.build_general_dashboard import generate_dashboard
from scripts.benchmark_visualizer import (
    compute_metrics, 
    plot_accuracy_dashboard, 
    OUTPUT_DASHBOARD_PATH, 
    ARTIFACT_DASHBOARD_PATHS
)

def main():
    json_path = "reports/benchmark_tests.json"
    csv_path = "reports/benchmark_tests.csv"
    monthly_csv_path = "reports/dashboard_monthly_records.csv"
    dow_csv_path = "reports/dashboard_dow_records.csv"
    
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # 1. Store Data Definitions
    cantu_entry = {
        "id": "pf00001305_coop_lombardia_cantu",
        "pf": "PF00001305",
        "store_name": "Coop Lombardia Cantù - CC Mirabello",
        "comune": "Cantù",
        "provincia": "CO",
        "address": "Via Lombardia, 68",
        "cap": "22063",
        "machine_type": "DAINT_G",
        "format": "Regional Hypermarket (7,200 m2, 34 checkouts)",
        "category": "Regional Hypermarket Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 978.8,
        "actual_bottles": 1030.0,
        "pred_users": 31.5,
        "actual_users": 37.0,
        "serviced_floor_bottles": 802.6,
        "p75_bottles": 1223.5,
        "p25_bottles": 802.6,
        "monthly_preds": [
            852.3, 853.2, 872.5, 945.9, 1014.5, 1096.4,
            1139.4, 1093.8, 1015.9, 938.2, 874.1, 858.5
        ],
        "monthly_actuals": [
            1008.1, 1068.5, 1083.9, 1078.2, 1148.7, 1045.1,
            1155.7, 1044.3, 1131.0, 1080.9, 953.4, 1006.2
        ],
        "monthly_preds_total": [
            26421, 23890, 27048, 28377, 31450, 32892,
            35321, 33908, 30477, 29084, 26223, 26614
        ],
        "monthly_actuals_total": [
            31252, 29918, 33602, 32346, 35610, 31354,
            35826, 32373, 33931, 33507, 28603, 31193
        ],
        "dow_preds": {
            "Monday": 979.7, "Tuesday": 950.7, "Wednesday": 952.9,
            "Thursday": 960.1, "Friday": 1002.0, "Saturday": 1147.0, "Sunday": 939.2
        }
    }

    portoferraio_entry = {
        "id": "pf00001460_unicoop_tirreno_portoferraio",
        "pf": "PF00001460",
        "store_name": "Unicoop Tirreno Portoferraio",
        "comune": "Portoferraio",
        "provincia": "LI",
        "address": "Via Tesei",
        "cap": "57037",
        "machine_type": "DAINT_G",
        "format": "Supermarket (1,355 m2, 10 checkouts)",
        "category": "Island Monopoly Commercial Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 1018.3,
        "actual_bottles": 1019.0,
        "pred_users": 31.6,
        "actual_users": 39.7,
        "serviced_floor_bottles": 865.8,
        "p75_bottles": 1173.4,
        "p25_bottles": 769.7,
        "monthly_preds": [
            865.8, 866.9, 872.5, 987.7, 1062.4, 1268.7,
            1385.5, 1330.0, 1107.5, 1020.8, 874.9, 858.8
        ],
        "monthly_actuals": [
            866.6, 865.7, 866.8, 970.4, 1059.3, 1165.2,
            1371.6, 1379.3, 1186.7, 1017.9, 902.2, 855.2
        ],
        "monthly_preds_total": [
            26840, 24273, 27048, 29631, 32934, 38061,
            42950, 41230, 33225, 31645, 26247, 26623
        ],
        "monthly_actuals_total": [
            26863, 24240, 26870, 29111, 32838, 34957,
            42518, 42759, 35602, 31556, 27065, 26511
        ],
        "dow_preds": {
            "Monday": 1069.4, "Tuesday": 1042.7, "Wednesday": 1035.7,
            "Thursday": 1036.2, "Friday": 1071.5, "Saturday": 1164.5, "Sunday": 955.8
        }
    }

    prato_entry = {
        "id": "pf00001636_unicoopfi_prato_parco",
        "pf": "PF00001636",
        "store_name": "Coop.fi - Prato (Parco Prato)",
        "comune": "Prato",
        "provincia": "PO",
        "address": "Via delle Pleiadi",
        "cap": "59100",
        "machine_type": "Ripet_C-front",
        "format": "Regional Mega-Hypermarket (5,139 m2, 38 checkouts)",
        "category": "National #1 High-Yield Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 4679.5,
        "actual_bottles": 5580.0,
        "pred_users": 98.5,
        "actual_users": 118.6,
        "serviced_floor_bottles": 3837.2,
        "p75_bottles": 5849.4,
        "p25_bottles": 3837.2,
        "monthly_preds": [
            4677.6, 4684.7, 4789.6, 4701.6, 4824.6, 4385.6,
            4346.0, 4264.6, 4747.5, 4627.7, 4778.2, 4642.6
        ],
        "monthly_actuals": [
            3540.3, 4864.0, 5102.3, 5848.8, 6248.0, 6991.3,
            7725.1, 6342.2, 7264.5, 6362.8, 5893.2, 5088.4
        ],
        "monthly_preds_total": [
            145006, 131172, 148478, 141048, 149563, 131568,
            134726, 132203, 142425, 143459, 143346, 143921
        ],
        "monthly_actuals_total": [
            109749, 136192, 158171, 175464, 193688, 209739,
            239478, 196608, 217935, 197247, 176796, 147564
        ],
        "dow_preds": {
            "Monday": 4651.3, "Tuesday": 4616.7, "Wednesday": 4624.2,
            "Thursday": 4617.1, "Friday": 4696.9, "Saturday": 4716.9, "Sunday": 4453.2
        }
    }

    cascina_entry = {
        "id": "mc1z14lfgrfg_coopfi_cascina",
        "pf": "MC1Z14LFGRFG",
        "store_name": "Coop.fi - Cascina",
        "comune": "Cascina",
        "provincia": "PI",
        "address": "Via Fosso Vecchio, 457",
        "cap": "56021",
        "machine_type": "Ripet_C-front",
        "format": "Commercial Superstore (4,714 m2, 26 checkouts)",
        "category": "Tier 1 - Strategic High-Volume Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 4303.7,
        "actual_bottles": 4520.0,
        "pred_users": 78.4,
        "actual_users": 76.5,
        "serviced_floor_bottles": 3043.8,
        "p75_bottles": 4640.0,
        "p25_bottles": 3043.8,
        "monthly_preds": [
            3921.1, 3922.9, 4150.3, 4222.3, 4498.5, 5003.8,
            5073.1, 5180.7, 4645.0, 4290.5, 4159.0, 3907.8
        ],
        "monthly_actuals": [
            4083.5, 3856.1, 4216.8, 4234.1, 4517.2, 4716.0,
            5370.6, 5182.9, 5265.9, 4630.6, 4545.5, 3998.3
        ],
        "monthly_preds_total": [
            121554, 109841, 128659, 126669, 139454, 150114,
            157266, 160602, 139350, 133006, 124770, 121142
        ],
        "monthly_actuals_total": [
            126589, 107971, 130721, 127023, 140033, 141480,
            166489, 160670, 157977, 143549, 136365, 123947
        ],
        "dow_preds": {
            "Monday": 4418.9, "Tuesday": 4418.5, "Wednesday": 4416.4,
            "Thursday": 4409.8, "Friday": 4447.7, "Saturday": 4870.3, "Sunday": 4276.5
        }
    }

    dimar_entry = {
        "id": "mc5epg7e0rzs_dimar_alessandria",
        "pf": "MC5EPG7E0RZS",
        "store_name": "Mercatò di Alessandria - Via Vecchia Torino",
        "comune": "Alessandria",
        "provincia": "AL",
        "address": "Via Vecchia Torino, 20",
        "cap": "15121",
        "machine_type": "Ripet_C-Ter",
        "format": "Superstore / Hypermarket (3,000 m2, 15 checkouts)",
        "category": "Commercial Retail Superstore",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 1502.5,
        "actual_bottles": 1780.0,
        "pred_users": 21.5,
        "actual_users": 34.5,
        "serviced_floor_bottles": 697.9,
        "p75_bottles": 1063.9,
        "p25_bottles": 697.9,
        "monthly_preds": [
            1392.8, 1392.8, 1459.9, 1450.1, 1536.1, 1616.5,
            1605.7, 1625.0, 1555.5, 1453.8, 1444.1, 1360.1
        ],
        "monthly_actuals": [
            1542.8, 1680.7, 1812.1, 1696.0, 1774.8, 2020.8,
            2010.7, 1944.6, 1956.4, 1697.5, 1617.4, 1545.1
        ],
        "monthly_preds_total": [
            43177, 38998, 45257, 43503, 47619, 48495,
            49777, 50375, 46665, 45068, 43323, 42163
        ],
        "monthly_actuals_total": [
            47827, 47060, 56175, 50880, 55019, 60624,
            62332, 60283, 58692, 52623, 48522, 47898
        ],
        "dow_preds": {
            "Monday": 1494.1, "Tuesday": 1474.3, "Wednesday": 1478.3,
            "Thursday": 1498.7, "Friday": 1561.0, "Saturday": 1795.2, "Sunday": 1396.2
        }
    }

    piccolo_entry = {
        "id": "pf00001201_piccolo_nola",
        "pf": "PF00001201",
        "store_name": "Piccolo di Nola - Via Variante",
        "comune": "Nola",
        "provincia": "NA",
        "address": "Via Variante 7 bis 64",
        "cap": "80035",
        "machine_type": "Ripet_C-front",
        "format": "Supermarket (1,600 m2, 5 checkouts)",
        "category": "Dense Urban Cluster Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 1609.4,
        "actual_bottles": 2020.0,
        "pred_users": 16.0,
        "actual_users": 44.5,
        "serviced_floor_bottles": 455.1,
        "p75_bottles": 693.8,
        "p25_bottles": 455.1,
        "monthly_preds": [
            1315.0, 1322.9, 1371.6, 1590.2, 1734.3, 1918.2,
            2005.1, 1986.1, 1825.5, 1629.2, 1379.0, 1311.5
        ],
        "monthly_actuals": [
            1761.1, 1831.0, 1715.9, 1828.7, 2095.7, 2303.3,
            2570.2, 2372.3, 2330.6, 2117.0, 1955.5, 1670.5
        ],
        "monthly_preds_total": [
            40765, 37041, 42520, 47706, 53763, 57546,
            62158, 61569, 54765, 50505, 41370, 40656
        ],
        "monthly_actuals_total": [
            54594, 51268, 53193, 54861, 64967, 69099,
            79676, 73541, 69918, 65627, 58665, 51786
        ],
        "dow_preds": {
            "Monday": 1629.2, "Tuesday": 1615.0, "Wednesday": 1611.3,
            "Thursday": 1615.8, "Friday": 1648.2, "Saturday": 1768.8, "Sunday": 1235.4
        }
    }

    conad_schio_entry = {
        "id": "pf00001367_conad_schio",
        "pf": "PF00001367",
        "store_name": "CONAD Schio - Via Vicenza",
        "comune": "Schio",
        "provincia": "VI",
        "address": "Via Vicenza, 30",
        "cap": "36015",
        "machine_type": "Ripet_C-front",
        "format": "Supermarket (1,270 m2, 6 checkouts)",
        "category": "Commercial Supermarket Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 646.0,
        "actual_bottles": 650.6,
        "pred_users": 17.7,
        "actual_users": 18.2,
        "serviced_floor_bottles": 468.1,
        "p75_bottles": 713.6,
        "p25_bottles": 468.1,
        "monthly_preds": [
            603.2, 602.8, 604.0, 614.2, 653.9, 773.9,
            848.0, 796.7, 681.1, 638.0, 618.1, 610.1
        ],
        "monthly_actuals": [
            566.8, 648.5, 661.7, 685.4, 672.6, 782.7,
            885.4, 736.8, 700.1, 634.7, 584.3, 563.2
        ],
        "monthly_preds_total": [
            18699, 16878, 18724, 18426, 20271, 23217,
            26288, 24698, 20433, 19778, 18543, 18913
        ],
        "monthly_actuals_total": [
            17571, 18158, 20513, 20562, 20851, 23481,
            27447, 22841, 21003, 19676, 17529, 17459
        ],
        "dow_preds": {
            "Monday": 682.3, "Tuesday": 660.8, "Wednesday": 657.3,
            "Thursday": 657.5, "Friday": 683.0, "Saturday": 855.5, "Sunday": 652.4
        }
    }

    citta_di_castello_entry = {
        "id": "pf00002687_coop_citta_di_castello",
        "pf": "PF00002687",
        "store_name": "Coop Città di Castello Le Fonti",
        "comune": "Città di Castello",
        "provincia": "PG",
        "address": "Via Scipione Lapi",
        "cap": "06012",
        "machine_type": "DAINT_G",
        "format": "Supermarket (1,449 m2, 8 checkouts)",
        "category": "Commercial Retail Supermarket Anchor",
        "clearance": "CONDITIONAL [REVIEW REQUIRED]",
        "pred_bottles": 710.2,
        "actual_bottles": 856.3,
        "pred_users": 14.0,
        "actual_users": 22.6,
        "serviced_floor_bottles": 582.4,
        "p75_bottles": 887.8,
        "p25_bottles": 582.4,
        "monthly_preds": [
            633.0, 633.7, 658.3, 702.2, 749.5, 865.8,
            868.4, 868.4, 758.5, 702.9, 652.9, 627.9
        ],
        "monthly_actuals": [
            711.3, 868.9, 1051.2, 1113.8, 911.3, 998.7,
            1018.2, 990.1, 1012.7, 829.3, 800.2, 729.0
        ],
        "monthly_preds_total": [
            19623, 17744, 20407, 21066, 23234, 25974,
            26920, 26920, 22755, 21790, 19587, 19465
        ],
        "monthly_actuals_total": [
            22050, 24329, 32587, 33414, 28250, 29961,
            31564, 30693, 30381, 25708, 24006, 22599
        ],
        "dow_preds": {
            "Monday": 741.4, "Tuesday": 727.2, "Wednesday": 724.2,
            "Thursday": 724.4, "Friday": 741.1, "Saturday": 883.6, "Sunday": 670.1
        }
    }

    fabrica_di_roma_entry = {
        "id": "pf00002527_coop_fabrica_di_roma",
        "pf": "PF00002527",
        "store_name": "Coop Fabrica di Roma - Via del Campo Sportivo",
        "comune": "Fabrica di Roma",
        "provincia": "VT",
        "address": "Via Del Campo Sportivo Snc",
        "cap": "01034",
        "machine_type": "DAINT_G",
        "format": "Supermarket (600 m2, 3 checkouts)",
        "category": "Uncontested Spatial Monopoly Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 1266.5,
        "actual_bottles": 1570.0,
        "pred_users": 16.9,
        "actual_users": 48.5,
        "serviced_floor_bottles": 1038.5,
        "p75_bottles": 1583.1,
        "p25_bottles": 1038.5,
        "monthly_preds": [
            1121.8, 1123.7, 1148.6, 1266.7, 1334.2, 1469.8,
            1554.6, 1514.0, 1355.1, 1268.7, 1129.1, 1111.6
        ],
        "monthly_actuals": [
            1347.2, 1463.7, 1487.7, 1555.5, 1508.7, 1836.2,
            1992.9, 2002.5, 1935.1, 1422.9, 1375.6, 1321.9
        ],
        "monthly_preds_total": [
            34776, 31464, 35607, 38001, 41360, 44094,
            48193, 46934, 40653, 39330, 33873, 34460
        ],
        "monthly_actuals_total": [
            41763, 40984, 46117, 46664, 46769, 55085,
            61781, 62076, 58054, 44111, 41268, 40980
        ],
        "dow_preds": {
            "Monday": 1309.3, "Tuesday": 1283.7, "Wednesday": 1280.1,
            "Thursday": 1283.5, "Friday": 1306.6, "Saturday": 1284.4, "Sunday": 980.6
        }
    }

    sesto_fiorentino_entry = {
        "id": "mcms8uec2qvh_coop_sesto_fiorentino",
        "pf": "MCMS8UEC2QVH",
        "store_name": "Coop.fi - Sesto Fiorentino (Centro*Sesto)",
        "comune": "Sesto Fiorentino",
        "provincia": "FI",
        "address": "Via Petrosa 19",
        "cap": "50019",
        "machine_type": "Ripet_C-front",
        "format": "Regional Hypermarket (5,305 m2, 40 checkouts)",
        "category": "National Tier 1 High-Yield Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 4962.4,
        "actual_bottles": 4330.0,
        "pred_users": 104.6,
        "actual_users": 107.5,
        "serviced_floor_bottles": 4125.7,
        "p75_bottles": 6461.5,
        "p25_bottles": 4125.7,
        "monthly_preds": [
            4371.8, 4372.9, 4812.7, 5193.2, 5545.4, 5589.4,
            5505.3, 5684.8, 5628.5, 5236.1, 4827.1, 4369.6
        ],
        "monthly_actuals": [
            3700.0, 4100.0, 4200.0, 4300.0, 5000.0, 5800.0,
            5600.0, 5200.0, 4400.0, 4300.0, 3800.0, 3600.0
        ],
        "monthly_preds_total": [
            135526, 122441, 149194, 155796, 171907, 167682,
            170664, 176229, 168855, 162319, 144813, 135458
        ],
        "monthly_actuals_total": [
            114700, 114800, 130200, 129000, 155000, 174000,
            173600, 161200, 132000, 133300, 114000, 111600
        ],
        "dow_preds": {
            "Monday": 5202.5, "Tuesday": 5097.6, "Wednesday": 5082.6,
            "Thursday": 5075.7, "Friday": 5200.6, "Saturday": 5078.8, "Sunday": 4235.8
        }
    }

    san_miniato_entry = {
        "id": "mcsm09lnt70o_sanminiato",
        "pf": "MCSM09LNT70O",
        "store_name": "Unicoop.Fi San Miniato",
        "comune": "San Miniato",
        "provincia": "PI",
        "address": "Via Ferrante Aporti",
        "cap": "56028",
        "machine_type": "Ripet_C-front",
        "format": "Superstore (1,996 m2, 23 checkouts)",
        "category": "Tier 1 - Strategic High-Volume Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 4012.1,
        "actual_bottles": 3990.0,
        "pred_users": 73.9,
        "actual_users": 54.3,
        "serviced_floor_bottles": 3289.9,
        "p75_bottles": 5015.1,
        "p25_bottles": 3410.3,
        "monthly_preds": [
            3525.8, 3527.1, 3805.7, 3936.8, 4287.2, 4811.7,
            4741.9, 4977.5, 4422.6, 3994.2, 3813.2, 3513.6
        ],
        "monthly_actuals": [
            2741.9, 2535.7, 3645.2, 3666.7, 3677.4, 4133.3,
            4387.1, 4032.3, 4100.0, 3871.0, 3733.3, 3354.8
        ],
        "monthly_preds_total": [
            109300, 98759, 117977, 118104, 132903, 144351,
            146999, 154302, 132678, 123820, 114396, 108922
        ],
        "monthly_actuals_total": [
            85000, 71000, 113000, 110000, 114000, 124000,
            136000, 125000, 123000, 120000, 112000, 104000
        ],
        "dow_preds": {
            "Monday": 4116.7, "Tuesday": 4116.9, "Wednesday": 4114.8,
            "Thursday": 4108.8, "Friday": 4152.0, "Saturday": 4640.3, "Sunday": 3986.4
        }
    }

    pistoia_entry = {
        "id": "pf00001535_coopfi_pistoia",
        "pf": "PF00001535",
        "store_name": "Coop.fi - Pistoia",
        "comune": "Pistoia",
        "provincia": "PT",
        "address": "Viale Macallè, 14",
        "cap": "51100",
        "machine_type": "Ripet_C-front",
        "format": "Regional Cooperative Supermarket (1,450 m2, 7 checkouts)",
        "category": "Regional Cooperative Supermarket Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 3490.9,
        "actual_bottles": 3340.0,
        "pred_users": 72.4,
        "actual_users": 98.5,
        "serviced_floor_bottles": 2862.5,
        "p75_bottles": 4363.6,
        "p25_bottles": 2862.5,
        "monthly_preds": [
            3138.4, 3155.5, 3291.1, 3530.4, 3383.5, 3367.7,
            3904.1, 3023.0, 3603.0, 3620.8, 3321.8, 3190.3
        ],
        "monthly_actuals": [
            2451.6, 2464.3, 3354.8, 3600.0, 3838.7, 4266.7,
            4548.4, 4064.5, 3957.4, 3200.0, 3100.0, 3000.0
        ],
        "monthly_preds_total": [
            97290, 88354, 102024, 105912, 104888, 101031,
            121027, 93713, 108090, 112245, 99654, 98899
        ],
        "monthly_actuals_total": [
            76000, 69000, 104000, 108000, 119000, 128000,
            141000, 126000, 118723, 99200, 93000, 93000
        ],
        "dow_preds": {
            "Monday": 3488.5, "Tuesday": 3387.8, "Wednesday": 3375.9,
            "Thursday": 3348.1, "Friday": 3449.3, "Saturday": 3931.3, "Sunday": 2788.7
        }
    }

    sesto_viale_pratese_entry = {
        "id": "pf00001941_coopfi_sesto_viale_pratese",
        "pf": "PF00001941",
        "store_name": "Coop.fi - Sesto Fiorentino - Viale Pratese",
        "comune": "Sesto Fiorentino",
        "provincia": "FI",
        "address": "Viale Pratese, 33/35",
        "cap": "50019",
        "machine_type": "Ripet_C-front",
        "format": "Regional Cooperative Supermarket (1,450 m2, 7 checkouts)",
        "category": "Regional Cooperative Supermarket Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 2768.0,
        "actual_bottles": 2660.0,
        "pred_users": 56.1,
        "actual_users": 80.1,
        "serviced_floor_bottles": 2269.8,
        "p75_bottles": 3460.0,
        "p25_bottles": 2269.8,
        "monthly_preds": [
            2465.4, 2466.1, 2692.2, 2764.1, 2905.7, 2988.8,
            2934.9, 3038.1, 2971.0, 2807.0, 2719.3, 2462.7
        ],
        "monthly_actuals": [
            2483.9, 2642.9, 2741.9, 2933.3, 2935.5, 3433.3,
            3741.9, 2967.7, 3333.3, 2741.9, 2666.7, 2419.4
        ],
        "monthly_preds_total": [
            76427, 69051, 83458, 82923, 90077, 89664,
            90982, 94181, 89130, 87017, 81579, 76344
        ],
        "monthly_actuals_total": [
            77000, 74000, 85000, 88000, 91000, 103000,
            116000, 92000, 100000, 85000, 80000, 75000
        ],
        "dow_preds": {
            "Monday": 2822.7, "Tuesday": 2769.6, "Wednesday": 2760.8,
            "Thursday": 2756.6, "Friday": 2821.1, "Saturday": 2887.1, "Sunday": 2354.2
        }
    }

    montevarchi_entry = {
        "id": "pf00002600_coop_montevarchi_levane",
        "pf": "PF00002600",
        "store_name": "Coop - Montevarchi (Levane)",
        "comune": "Montevarchi",
        "provincia": "AR",
        "address": "Piazza Falcone e Borsellino 3",
        "cap": "52025",
        "machine_type": "DAINT_G",
        "format": "Neighborhood Supermarket (550 m2, 4 checkouts)",
        "category": "Uncontested Municipal Cooperative Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 650.1,
        "actual_bottles": 672.8,
        "pred_users": 18.1,
        "actual_users": 21.2,
        "serviced_floor_bottles": 552.6,
        "p75_bottles": 747.6,
        "p25_bottles": 552.6,
        "monthly_preds": [
            643.2, 642.7, 652.7, 615.1, 607.6, 675.6,
            710.9, 689.3, 622.6, 624.6, 654.8, 639.6
        ],
        "monthly_actuals": [
            670.7, 557.4, 712.2, 714.8, 587.1, 787.4,
            777.6, 586.3, 655.1, 569.0, 649.6, 589.9
        ],
        "monthly_preds_total": [
            19939, 17996, 20234, 18453, 18836, 20268,
            22038, 21368, 18678, 19363, 19644, 19828
        ],
        "monthly_actuals_total": [
            20793, 15608, 22077, 21444, 18201, 23623,
            24105, 18175, 19652, 17638, 19487, 18287
        ],
        "dow_preds": {
            "Monday": 661.8, "Tuesday": 637.5, "Wednesday": 637.1,
            "Thursday": 640.4, "Friday": 679.0, "Saturday": 750.0, "Sunday": 641.1
        }
    }

    conad_aci_santantonio_entry = {
        "id": "pf00002012_conad_aci_santantonio",
        "pf": "PF00002012",
        "store_name": "CONAD - Aci Sant'Antonio",
        "comune": "Aci Sant'Antonio",
        "provincia": "CT",
        "address": "Via Marchesana 50",
        "cap": "95025",
        "machine_type": "Ripet_C-front",
        "format": "Supermarket (2,000 m2, 7 checkouts)",
        "category": "High-Volume Urban Commercial Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 1830.8,
        "actual_bottles": 1820.0,
        "pred_users": 41.0,
        "actual_users": 37.3,
        "serviced_floor_bottles": 1556.2,
        "p75_bottles": 2105.4,
        "p25_bottles": 1556.2,
        "monthly_preds": [
            1592.3, 1594.2, 1638.6, 1957.9, 2167.6, 2397.9,
            2419.6, 2471.3, 2191.2, 1951.0, 1623.9, 1585.1
        ],
        "monthly_actuals": [
            1609.1, 1709.9, 1605.7, 1741.0, 1659.1, 2145.2,
            2292.2, 2258.1, 2400.0, 2115.1, 1925.1, 1562.4
        ],
        "monthly_preds_total": [
            49361, 44638, 50797, 58737, 67196, 71937,
            75008, 76610, 65736, 60481, 48717, 49138
        ],
        "monthly_actuals_total": [
            49882, 47878, 49778, 52230, 51432, 64357,
            71058, 70001, 72000, 65568, 57753, 48434
        ],
        "dow_preds": {
            "Monday": 1978.1, "Tuesday": 1954.4, "Wednesday": 1947.9,
            "Thursday": 1949.0, "Friday": 1971.3, "Saturday": 1965.4, "Sunday": 1480.0
        }
    }

    # Remove existing instances of these stores if present
    target_pfs = {"PF00001305", "PF00001460", "PF00001636", "MC1Z14LFGRFG", "MC5EPG7E0RZS", "PF00001201", "PF00001367", "PF00002687", "PF00002527", "MCMS8UEC2QVH", "MCSM09LNT70O", "PF00001535", "PF00001941", "PF00002600", "PF00002012"}
    benchmarks = [b for b in benchmarks if b.get("pf") not in target_pfs]
    
    # Append updated and new entries
    benchmarks.append(cantu_entry)
    benchmarks.append(portoferraio_entry)
    benchmarks.append(prato_entry)
    benchmarks.append(cascina_entry)
    benchmarks.append(dimar_entry)
    benchmarks.append(piccolo_entry)
    benchmarks.append(conad_schio_entry)
    benchmarks.append(citta_di_castello_entry)
    benchmarks.append(fabrica_di_roma_entry)
    benchmarks.append(sesto_fiorentino_entry)
    benchmarks.append(san_miniato_entry)
    benchmarks.append(pistoia_entry)
    benchmarks.append(sesto_viale_pratese_entry)
    benchmarks.append(montevarchi_entry)
    benchmarks.append(conad_aci_santantonio_entry)
    
    # Save benchmark_tests.json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    print(f"Updated {json_path} (Total stores: {len(benchmarks)})")
    
    # 2. Update benchmark_tests.csv
    csv_rows = []
    for b in benchmarks:
        pb = b["pred_bottles"]
        ab = b["actual_bottles"]
        diff_b = round(pb - ab, 1)
        pct_b = round((pb - ab) / ab * 100.0, 1) if ab > 0 else 0.0
        acc_b = round(max(0.0, 100.0 - abs(pct_b)), 1)
        
        pu = b["pred_users"]
        au = b["actual_users"]
        diff_u = round(pu - au, 1)
        pct_u = round((pu - au) / au * 100.0, 1) if au > 0 else 0.0
        acc_u = round(max(0.0, 100.0 - abs(pct_u)), 1)
        
        csv_rows.append({
            "store_name": b["store_name"],
            "comune": b["comune"],
            "provincia": b["provincia"],
            "category": b["category"],
            "pred_bottles": pb,
            "actual_bottles": ab,
            "error_bottles": diff_b,
            "pct_error_bottles": pct_b,
            "accuracy_bottles": acc_b,
            "pred_users": pu,
            "actual_users": au,
            "error_users": diff_u,
            "pct_error_users": pct_u,
            "accuracy_users": acc_u
        })
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {len(df_csv)} benchmark rows.")
    
    # 3. Update dashboard_monthly_records.csv
    if os.path.exists(monthly_csv_path):
        df_monthly = pd.read_csv(monthly_csv_path)
        df_monthly = df_monthly[~df_monthly["punto_fisico"].isin(target_pfs)]
    else:
        df_monthly = pd.DataFrame()
        
    month_names = [
        "01_Gennaio", "02_Febbraio", "03_Marzo", "04_Aprile", "05_Maggio", "06_Giugno",
        "07_Luglio", "08_Agosto", "09_Settembre", "10_Ottobre", "11_Novembre", "12_Dicembre"
    ]
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    contexts = [
        "Winter baseline / off-peak", "Short winter cycle", "Early spring resumption",
        "Spring baseline", "Late spring ramp-up", "Early summer hydration ramp",
        "Summer peak hydration", "Mid-summer holiday rhythm", "School / work return",
        "Autumn steady-state", "Late autumn transition", "Holiday shopping period"
    ]
    
    new_monthly_rows = []
    for entry in [cantu_entry, portoferraio_entry, prato_entry, cascina_entry, dimar_entry, piccolo_entry, conad_schio_entry, citta_di_castello_entry, fabrica_di_roma_entry, sesto_fiorentino_entry, san_miniato_entry, pistoia_entry, sesto_viale_pratese_entry, montevarchi_entry, conad_aci_santantonio_entry]:
        pu_base = entry["pred_users"]
        au_base = entry["actual_users"]
        for m_idx in range(12):
            pb_day = entry["monthly_preds"][m_idx]
            ab_day = entry["monthly_actuals"][m_idx]
            days = days_in_months[m_idx]
            pb_tot = entry["monthly_preds_total"][m_idx]
            ab_tot = entry["monthly_actuals_total"][m_idx]
            
            diff_day = round(pb_day - ab_day, 1)
            pct_err = round((pb_day - ab_day) / ab_day * 100.0, 1)
            acc_pct = round(max(0.0, 100.0 - abs(pct_err)), 1)
            
            new_monthly_rows.append({
                "store_name": entry["store_name"],
                "punto_fisico": entry["pf"],
                "comune": entry["comune"],
                "provincia": entry["provincia"],
                "month_num": m_idx + 1,
                "month_name": month_names[m_idx],
                "days_in_month": days,
                "pred_bottles_day": pb_day,
                "actual_bottles_day": ab_day,
                "diff_bottles_day": diff_day,
                "pct_error_bottles": pct_err,
                "accuracy_bottles_pct": acc_pct,
                "pred_monthly_total": pb_tot,
                "actual_monthly_total": ab_tot,
                "pred_presenze_day": pu_base,
                "actual_presenze_day": au_base,
                "pred_presenze_total": round(pu_base * days),
                "actual_presenze_total": round(au_base * days),
                "operational_context": contexts[m_idx]
            })
            
    df_new_monthly = pd.DataFrame(new_monthly_rows)
    df_combined_monthly = pd.concat([df_monthly, df_new_monthly], ignore_index=True)
    df_combined_monthly.to_csv(monthly_csv_path, index=False)
    print(f"Updated {monthly_csv_path} with {len(df_combined_monthly)} total monthly records.")
    
    # 4. Update dashboard_dow_records.csv
    if os.path.exists(dow_csv_path):
        df_dow = pd.read_csv(dow_csv_path)
        df_dow = df_dow[~df_dow["punto_fisico"].isin(target_pfs)]
    else:
        df_dow = pd.DataFrame()
        
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_contexts = {
        "Monday": "Weekday baseline", "Tuesday": "Weekday baseline", "Wednesday": "Weekday baseline",
        "Thursday": "Weekday baseline", "Friday": "Pre-weekend shopping ramp",
        "Saturday": "Peak Saturday family grocery trip", "Sunday": "Sunday shopping"
    }
    
    new_dow_rows = []
    for entry in [cantu_entry, portoferraio_entry, prato_entry, cascina_entry, dimar_entry, piccolo_entry, conad_schio_entry, citta_di_castello_entry, fabrica_di_roma_entry, sesto_fiorentino_entry, san_miniato_entry, pistoia_entry, sesto_viale_pratese_entry, montevarchi_entry, conad_aci_santantonio_entry]:
        d_preds = entry.get("dow_preds", {})
        for day in dow_order:
            val = d_preds.get(day, entry["pred_bottles"])
            new_dow_rows.append({
                "store_name": entry["store_name"],
                "punto_fisico": entry["pf"],
                "day_of_week": day,
                "pred_bottles_mean": val,
                "pred_bottles_median": val,
                "operational_context": dow_contexts[day]
            })
            
    df_new_dow = pd.DataFrame(new_dow_rows)
    df_combined_dow = pd.concat([df_dow, df_new_dow], ignore_index=True)
    df_combined_dow.to_csv(dow_csv_path, index=False)
    print(f"Updated {dow_csv_path} with {len(df_combined_dow)} total day-of-week records.")
    
    # 5. Generate accuracy_dashboard.html
    print("Rebuilding reports/accuracy_dashboard.html...")
    generate_dashboard()
    print("Successfully rebuilt reports/accuracy_dashboard.html")
    
    # 6. Rebuild model_accuracy_dashboard.png
    print("Rebuilding model_accuracy_dashboard.png...")
    results = compute_metrics(benchmarks)
    plot_accuracy_dashboard(results, [OUTPUT_DASHBOARD_PATH] + ARTIFACT_DASHBOARD_PATHS)
    print("Successfully rebuilt and saved model_accuracy_dashboard.png to reports and artifacts.")
    
    # 7. Print summary metrics
    preds = np.array([r["pred_bottles"] for r in results])
    acts = np.array([r["actual_bottles"] for r in results])
    accuracies = np.array([r["accuracy_bottles"] for r in results])
    mae = np.mean(np.abs(preds - acts))
    wape = (np.sum(np.abs(preds - acts)) / np.sum(acts)) * 100.0
    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)
    
    print("\n" + "=" * 65)
    print(f"DASHBOARD UPDATE COMPLETE — {len(benchmarks)} BENCHMARK STORES ACTIVE")
    print(f"Network Mean Accuracy:   {mean_acc:.1f}%")
    print(f"Network Median Accuracy: {median_acc:.1f}%")
    print(f"Weighted APE (WAPE):     {wape:.1f}%")
    print(f"Mean Absolute Error:     {mae:.1f} bottles/day")
    print("=" * 65)

if __name__ == "__main__":
    main()
