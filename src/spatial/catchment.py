"""
Spatial Catchment & Competitive Density Analyzer for prospective retail locations.
Implements:
1. Fast geodesic Haversine distance calculations against all 1,657 operating Coripet PFs.
2. Nearest Coripet RVM detection, distance (km), and municipal network density.
3. Network position & cannibalization risk detection (< 500m).
4. Competitive retail grocery density from Nielsen store universe (5,703 localized catchments).
5. Optional OpenStreetMap Overpass live API connector with safe offline fallback.
"""
import os
import json
import re
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

def haversine_np(lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance calculation in kilometers."""
    r_lat1 = np.radians(lat1)
    r_lon1 = np.radians(lon1)
    r_lat2 = np.radians(lat2_arr)
    r_lon2 = np.radians(lon2_arr)

    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c  # Earth radius in km

def extract_street_core(address_str: str) -> str:
    """Extracts core street name (e.g. 'VIA FERROVIA 66' -> 'FERROVIA')."""
    norm = re.sub(r"[^\w\s]", " ", str(address_str).upper())
    prefixes = ["VIA", "CORSO", "PIAZZA", "VIALE", "VICOLO", "STRADA", "LOC", "LOCALITA", "LARGO"]
    tokens = norm.split()
    filtered = [tok for tok in tokens if tok not in prefixes and not tok.isdigit()]
    return " ".join(filtered) if filtered else norm.strip()

class SpatialCatchmentAnalyzer:
    """
    Evaluates spatial network cannibalization, trade area density, 
    and competitive retail environment for prospective Coripet RVM points.
    """
    def __init__(
        self,
        network_path: str = "data/processed/coripet_rvm_spatial_network.parquet",
        density_cap_path: str = "data/processed/competitive_density_lookup.parquet",
        density_comune_path: str = "data/processed/competitive_density_comune.parquet"
    ):
        self.network_loaded = False
        self.density_loaded = False
        
        # 1. Load Coripet Network Registry
        if os.path.exists(network_path):
            self.network_df = pd.read_parquet(network_path)
            self.pfs_lat = self.network_df["Latitudine"].to_numpy(dtype=float)
            self.pfs_lon = self.network_df["Longitudine"].to_numpy(dtype=float)
            self.pfs_id = self.network_df["PF"].to_numpy(dtype=str)
            self.pfs_client = self.network_df["CLIENTE"].fillna("UNKNOWN").to_numpy(dtype=str)
            self.pfs_comune = self.network_df["Comune"].fillna("UNKNOWN").to_numpy(dtype=str)
            self.pfs_cap = self.network_df["Cap"].fillna("").astype(str).to_numpy(dtype=str)
            self.pfs_address = self.network_df["Indirizzo"].fillna("").to_numpy(dtype=str)
            self.pfs_bottles = self.network_df["mean_bottiglie_day"].to_numpy(dtype=float)
            self.pfs_presenze = (
                self.network_df["mean_presenze_day"].to_numpy(dtype=float)
                if "mean_presenze_day" in self.network_df.columns
                else np.zeros(len(self.network_df), dtype=float)
            )
            self.pfs_machine_type = (
                self.network_df["TipoMacchina"].fillna("UNKNOWN").to_numpy(dtype=str)
                if "TipoMacchina" in self.network_df.columns
                else np.array(["UNKNOWN"] * len(self.network_df), dtype=str)
            )
            self.network_loaded = True
        else:
            self.network_df = pd.DataFrame()
            
        # 2. Load Competitive Density Indices
        self.density_cap_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if os.path.exists(density_cap_path):
            df_cap = pd.read_parquet(density_cap_path)
            for _, r in df_cap.iterrows():
                key = (str(r["CAP_str"]).strip().zfill(5), str(r["COMUNE_norm"]).strip().upper())
                self.density_cap_dict[key] = {
                    "total_grocery_stores": int(r["total_grocery_stores"]),
                    "supermarkets": int(r["supermarkets"]),
                    "hypermarkets": int(r["hypermarkets"]),
                    "discounts": int(r["discounts"]),
                    "neighborhood_stores": int(r["neighborhood_stores"]),
                    "shopping_centers": int(r["shopping_centers"]),
                    "top_competing_banners": str(r["top_competing_banners"]),
                    "density_tier": str(r["density_tier"])
                }
                
        self.density_comune_dict: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(density_comune_path):
            df_com = pd.read_parquet(density_comune_path)
            for _, r in df_com.iterrows():
                key = str(r["COMUNE_norm"]).strip().upper()
                self.density_comune_dict[key] = {
                    "total_grocery_stores": int(r["total_grocery_stores"]),
                    "supermarkets": int(r["supermarkets"]),
                    "hypermarkets": int(r["hypermarkets"]),
                    "discounts": int(r["discounts"]),
                    "neighborhood_stores": int(r["neighborhood_stores"]),
                    "shopping_centers": int(r["shopping_centers"]),
                    "top_competing_banners": str(r["top_competing_banners"]),
                    "density_tier": str(r["density_tier"])
                }
            self.density_loaded = True

    def analyze_coripet_network(
        self, 
        lat: float, 
        lon: float, 
        cap: str, 
        comune: str,
        address: str = "",
        surface: float = 25.0,
        store_name: str = "",
        exclude_self_benchmark: bool = True
    ) -> Dict[str, Any]:
        """
        Computes distance to all 1,657 Coripet machines, identifies the nearest point,
        and diagnoses network positioning (Monopoly vs. Cannibalization vs. Cluster).
        
        If exclude_self_benchmark=True:
        Detects if this query is benchmarking an already-installed machine on site (< 80m with
        matching banner or street), excludes that self-machine from neighboring cannibalization calculations,
        and evaluates the store as a standalone unit against the nearest neighboring machine.
        """
        if not self.network_loaded or len(self.pfs_lat) == 0:
            return {
                "nearest_coripet_km": 999.0,
                "nearest_pf_id": "N/A",
                "nearest_pf_name": "N/A",
                "nearest_pf_comune": "N/A",
                "nearest_pf_address": "N/A",
                "coripet_count_1km": 0,
                "coripet_count_3km": 0,
                "coripet_count_5km": 0,
                "coripet_count_10km": 0,
                "same_municipality_count": 0,
                "network_position": "UNCONTESTED_MONOPOLY",
                "network_status_description": "First Coripet installation in area.",
                "is_benchmark_self_match": False,
                "benchmark_pf_id": None,
                "benchmark_pf_name": None,
                "benchmark_pf_address": None,
                "benchmark_pf_hist_bottles": None
            }

        # Vectorized geodesic distance to all PFs
        distances = haversine_np(lat, lon, self.pfs_lat, self.pfs_lon)
        user_street_core = extract_street_core(address)
        adj_distances = distances.copy()

        # 1. Detect if this query is an installed site benchmark (Self-Match Exclusion)
        self_match_idx = None
        if exclude_self_benchmark and len(self.pfs_lat) > 0:
            user_brand = re.sub(r"[^\w\s]", " ", str(store_name).upper()).strip()
            # Check machines within the immediate physical site footprint (< 80 meters)
            cand_mask = (distances < 0.08)
            if np.any(cand_mask):
                for c_idx in np.where(cand_mask)[0]:
                    pf_client = str(self.pfs_client[c_idx]).strip().upper()
                    pf_street = extract_street_core(self.pfs_address[c_idx])
                    
                    # Verify brand identity match
                    brand_matched = bool(user_brand and (
                        user_brand in pf_client or pf_client in user_brand or
                        ("ESSELUNGA" in user_brand and "ESSELUNGA" in pf_client) or
                        ("CONAD" in user_brand and "CONAD" in pf_client) or
                        ("COOP" in user_brand and "COOP" in pf_client) or
                        ("CARREFOUR" in user_brand and "CARREFOUR" in pf_client) or
                        ("LEROY" in user_brand and "LEROY" in pf_client) or
                        ("BRICO" in user_brand and "BRICO" in pf_client) or
                        ("TECNOMAT" in user_brand and "TECNOMAT" in pf_client) or
                        ("DECATHLON" in user_brand and "DECATHLON" in pf_client) or
                        ("BENNET" in user_brand and "BENNET" in pf_client) or
                        ("PAM" in user_brand and "PAM" in pf_client) or
                        ("TIGROS" in user_brand and "TIGROS" in pf_client) or
                        ("IPER" in user_brand and "IPER" in pf_client) or
                        ("DESPAR" in user_brand and "DESPAR" in pf_client) or
                        ("EUROSPIN" in user_brand and "EUROSPIN" in pf_client) or
                        ("LIDL" in user_brand and "LIDL" in pf_client) or
                        ("ALDI" in user_brand and "ALDI" in pf_client) or
                        ("PENNY" in user_brand and "PENNY" in pf_client) or
                        ("MD" in user_brand and "MD" in pf_client)
                    ))
                    # Verify street address match
                    street_matched = bool(user_street_core and pf_street and (
                        user_street_core in pf_street or pf_street in user_street_core
                    ))
                    
                    # If user specified a store/brand name, it MUST match the installed client
                    # to prevent false centroid collisions with unrelated retail banners
                    is_real_match = False
                    if user_brand:
                        if brand_matched and (street_matched or distances[c_idx] < 0.05):
                            is_real_match = True
                    else:
                        if street_matched and distances[c_idx] < 0.05:
                            is_real_match = True

                    if is_real_match:
                        self_match_idx = int(c_idx)
                        break

        # Check street relationship and adjust false municipal centroid co-locations
        if address and user_street_core:
            # Check all machines with raw distance < 0.5 km (potential centroid collision)
            close_mask = (adj_distances < 0.5)
            if self_match_idx is not None:
                close_mask[self_match_idx] = False  # Keep self at true distance, don't artificially displace
            if np.any(close_mask):
                inter_street_est = max(1.2, min(4.5, round(0.4 * np.sqrt(max(4.0, surface)), 1)))
                for idx in np.where(close_mask)[0]:
                    pf_street_core = extract_street_core(self.pfs_address[idx])
                    is_same_street = bool(pf_street_core and (user_street_core in pf_street_core or pf_street_core in user_street_core))
                    if not is_same_street:
                        adj_distances[idx] = float(inter_street_est)

        # Build candidate evaluation set (excluding self-match if benchmarking active installation)
        if self_match_idx is not None and len(adj_distances) > 1:
            eval_mask = np.ones(len(adj_distances), dtype=bool)
            eval_mask[self_match_idx] = False
            eval_indices = np.where(eval_mask)[0]
        else:
            eval_indices = np.arange(len(adj_distances))

        # Sort candidate indices based on adjusted distances
        sorted_eval = eval_indices[np.argsort(adj_distances[eval_indices])]
        nearest_idx = int(sorted_eval[0])
        nearest_dist = float(adj_distances[nearest_idx])
        
        # Municipal counts (case-insensitive) among evaluated candidate machines
        clean_comune = str(comune).strip().upper()
        pfs_comune_upper = np.char.upper(self.pfs_comune[eval_indices])
        same_comune_mask = np.char.find(pfs_comune_upper, clean_comune) >= 0 if clean_comune else np.zeros(len(eval_indices), dtype=bool)
        same_muni_count = int(np.sum(same_comune_mask))

        # Recalculate counts within radius thresholds strictly from evaluated candidate machines
        count_1km = int(np.sum(adj_distances[eval_indices] <= 1.0))
        count_3km = int(np.sum(adj_distances[eval_indices] <= 3.0))
        count_5km = int(np.sum(adj_distances[eval_indices] <= 5.0))
        count_10km = int(np.sum(adj_distances[eval_indices] <= 10.0))

        # Enforce strict geometric consistency:
        if nearest_dist > 1.0:
            count_1km = 0
        if nearest_dist > 3.0:
            count_3km = 0
        if nearest_dist > 5.0:
            count_5km = 0

        # Classify network positioning
        if nearest_dist < 0.5:
            pos = "CANNIBALIZATION_ALERT"
            desc = f"Existing machine ({self.pfs_client[nearest_idx]}) is only {nearest_dist*1000:.0f}m away on {self.pfs_address[nearest_idx]}. High risk of local volume dilution."
        elif count_5km == 0:
            pos = "UNCONTESTED_MONOPOLY"
            desc = "Sole Coripet recycling destination within 5+ km. Maximum localized catchment capture."
        elif count_5km <= 2:
            pos = "EXPANSION_NODE"
            desc = f"Healthy network density ({count_5km} machines within 5 km). Nearest machine is {nearest_dist:.1f} km away on {self.pfs_address[nearest_idx]}."
        else:
            pos = "DENSE_CLUSTER"
            desc = f"Mature urban network cluster ({count_5km} machines within 5 km). High public brand awareness."

        if self_match_idx is not None:
            desc = f"Benchmarking existing active installation ({self.pfs_client[self_match_idx]} - {self.pfs_id[self_match_idx]}). {desc}"

        return {
            "nearest_coripet_km": round(nearest_dist, 2),
            "nearest_pf_id": str(self.pfs_id[nearest_idx]),
            "nearest_pf_name": str(self.pfs_client[nearest_idx]),
            "nearest_pf_comune": str(self.pfs_comune[nearest_idx]),
            "nearest_pf_address": str(self.pfs_address[nearest_idx]),
            "nearest_pf_hist_bottles": round(float(self.pfs_bottles[nearest_idx]), 1),
            "nearest_pf_hist_presenze": round(float(self.pfs_presenze[nearest_idx]), 1) if self.pfs_presenze is not None and self.pfs_presenze[nearest_idx] > 0 else None,
            "coripet_count_1km": count_1km,
            "coripet_count_3km": count_3km,
            "coripet_count_5km": count_5km,
            "coripet_count_10km": count_10km,
            "same_municipality_count": same_muni_count,
            "network_position": pos,
            "network_status_description": desc,
            "is_benchmark_self_match": self_match_idx is not None,
            "benchmark_pf_id": str(self.pfs_id[self_match_idx]) if self_match_idx is not None else None,
            "benchmark_pf_name": str(self.pfs_client[self_match_idx]) if self_match_idx is not None else None,
            "benchmark_pf_address": str(self.pfs_address[self_match_idx]) if self_match_idx is not None else None,
            "benchmark_pf_machine_type": str(self.pfs_machine_type[self_match_idx]) if self_match_idx is not None else None,
            "benchmark_pf_hist_bottles": round(float(self.pfs_bottles[self_match_idx]), 1) if self_match_idx is not None else None,
            "benchmark_pf_hist_presenze": round(float(self.pfs_presenze[self_match_idx]), 1) if self_match_idx is not None and self.pfs_presenze[self_match_idx] > 0 else None
        }

    def analyze_competitive_density(self, cap: str, comune: str) -> Dict[str, Any]:
        """
        Retrieves grocery retail density and format composition from Nielsen's 25,104 store universe.
        """
        clean_cap = str(cap).strip().zfill(5)
        clean_comune = str(comune).strip().upper()
        
        # 1. Exact (CAP, Comune) lookup
        if (clean_cap, clean_comune) in self.density_cap_dict:
            res = self.density_cap_dict[(clean_cap, clean_comune)].copy()
            res["lookup_level"] = "LOCAL_POSTAL_DISTRICT (CAP)"
            return res
            
        # 2. Comune-level lookup
        if clean_comune in self.density_comune_dict:
            res = self.density_comune_dict[clean_comune].copy()
            res["lookup_level"] = "MUNICIPALITY_WIDE"
            return res
            
        # 3. Fuzzy Comune match
        for c_key, data in self.density_comune_dict.items():
            if len(c_key) >= 4 and (c_key in clean_comune or clean_comune in c_key):
                res = data.copy()
                res["lookup_level"] = "MUNICIPALITY_FUZZY"
                return res

        # 4. National benchmark fallback
        return {
            "total_grocery_stores": 4,
            "supermarkets": 2,
            "hypermarkets": 0,
            "discounts": 1,
            "neighborhood_stores": 1,
            "shopping_centers": 0,
            "top_competing_banners": "INDIPENDENTI, CONAD, COOP",
            "density_tier": "MODERATE_COMMERCIAL_CORRIDOR",
            "lookup_level": "NATIONAL_MEDIAN_FALLBACK"
        }

    def fetch_osm_amenities(self, lat: float, lon: float, radius_m: int = 1000) -> Dict[str, Any]:
        """
        Queries OpenStreetMap Overpass API for transit hubs and commercial amenities within 1 km.
        Safely wraps requests with strict timeout and offline fallback.
        """
        query = f"""
        [out:json][timeout:2];
        (
          node["highway"="bus_stop"](around:{radius_m},{lat},{lon});
          node["railway"="station"](around:{radius_m},{lat},{lon});
          node["shop"="supermarket"](around:{radius_m},{lat},{lon});
        );
        out count;
        """
        try:
            url = "https://overpass-api.de/api/interpreter"
            data = urllib.parse.urlencode({"data": query}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "CoripetPredictor/4.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                content = json.loads(resp.read().decode("utf-8"))
                counts = content.get("elements", [{}])[0].get("tags", {})
                total_nodes = int(counts.get("total", 0))
                return {
                    "osm_status": "ONLINE_QUERY_SUCCESS",
                    "transit_and_retail_nodes_1km": total_nodes,
                    "commercial_activity_level": "HIGH" if total_nodes >= 10 else ("MODERATE" if total_nodes >= 3 else "QUIET")
                }
        except Exception:
            return {
                "osm_status": "OFFLINE_FALLBACK",
                "transit_and_retail_nodes_1km": "N/A (Offline mode)",
                "commercial_activity_level": "CALCULATED_FROM_NIELSEN"
            }

    def analyze_catchment(
        self, 
        lat: float, 
        lon: float, 
        cap: str, 
        comune: str,
        address: str = "",
        surface: float = 25.0,
        store_name: str = "",
        exclude_self_benchmark: bool = True,
        enable_osm: bool = False
    ) -> Dict[str, Any]:
        """End-to-end spatial catchment, network distance, and competitor density evaluation."""
        net_info = self.analyze_coripet_network(
            lat, lon, cap, comune, 
            address=address, 
            surface=surface,
            store_name=store_name,
            exclude_self_benchmark=exclude_self_benchmark
        )
        comp_info = self.analyze_competitive_density(cap, comune)
        
        osm_info = {}
        if enable_osm:
            osm_info = self.fetch_osm_amenities(lat, lon)
        else:
            osm_info = {
                "osm_status": "DISABLED_FAST_MODE",
                "transit_and_retail_nodes_1km": "NOT_QUERIED",
                "commercial_activity_level": comp_info["density_tier"]
            }
            
        return {
            "coripet_network": net_info,
            "competitive_density": comp_info,
            "osm_amenities": osm_info
        }
