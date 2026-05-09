#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse Complicaciones Column - Extract Clinical Variables
=========================================================

This script parses the 'complicaciones' column from the RECUIMA dataset
to extract ONLY the clinical variables that are embedded in the text.

The extracted variables are binary indicators for:
- FV (Fibrilación Ventricular)
- TV (Taquicardia Ventricular)  
- BAV alto grado (Bloqueo AV de alto grado: 2do Mobitz II, 3er grado, 2:1)
- BAV bajo grado (Bloqueo AV de 1er grado)
- Shock cardiogénico
- Otras arritmias
- Complicaciones mecánicas
- Otras complicaciones clínicas

NOTE: This script does NOT add derived variables (like edad > 70, TAS < 100, etc.)
Those calculations should be done at scoring time, not stored in the dataset.

Author: ML Team
Date: 2026-01-07
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN DEFINITIONS FROM 'complicaciones' COLUMN
# =============================================================================

# Tokens for Ventricular Fibrillation (FV)
FV_TOKENS = {'fv', 'pcr-fv'}

# Tokens for Ventricular Tachycardia (TV)
TV_TOKENS = {'tv'}

# Tokens for High-Grade AV Block (BAV 2nd Mobitz II, 3rd degree, 2:1)
# These are the ones that count for RECUIMA (1 point)
BAV_ALTO_GRADO_TOKENS = {'bav3', 'bav2:1', 'bav2', 'bav', 'pcr-bav-pericar'}

# Tokens for Low-Grade AV Block (1st degree) - Does NOT count for RECUIMA
BAV_BAJO_GRADO_TOKENS = {'bav1'}

# Shock cardiogénico tokens
SHOCK_TOKENS = {'shock', 'shock_c'}

# Fallo de bomba / Insuficiencia cardíaca
IC_TOKENS = {'fallo_bomba', 'congpulm', 'eap'}

# Paro cardiorrespiratorio
PCR_TOKENS = {'pcr', 'pcr-fv', 'pcr-bav-pericar', 'asist'}

# Otras arritmias (no FV/TV)
OTRAS_ARRITMIAS_TOKENS = {'farvr', 'farva', 'farvl', 'arritmia', 'ts', 'bs'}

# Complicaciones mecánicas
COMP_MECANICAS_TOKENS = {
    'rupt p lib', 'rupt tiv', 'rupt muscp', 'ruptura',
    'aneurap', 'pseudoaneur', 'taponam', 'insmit', 'i.mitral'
}


def parse_complicaciones(text: str) -> Dict[str, bool]:
    """
    Parse a complicaciones string and return binary flags.
    
    Args:
        text: The complicaciones text field (e.g., "tv;bav3;shock")
        
    Returns:
        Dictionary with binary flags for each complication type
    """
    result = {
        # Variables for RECUIMA scale (parsed from complicaciones)
        'comp_fv': False,               # Fibrilación ventricular
        'comp_tv': False,               # Taquicardia ventricular
        'comp_fv_tv': False,            # FV or TV (for RECUIMA: 2 points)
        'comp_bav_alto_grado': False,   # BAV alto grado (for RECUIMA: 1 point)
        'comp_bav_bajo_grado': False,   # BAV de 1er grado (NOT for RECUIMA)
        
        # Other important complications (for analysis/other uses)
        'comp_shock': False,            # Shock cardiogénico
        'comp_ic': False,               # Insuficiencia cardíaca / fallo bomba
        'comp_pcr': False,              # Paro cardiorrespiratorio
        'comp_otras_arritmias': False,  # Otras arritmias (FA, flutter, etc.)
        'comp_mecanicas': False,        # Complicaciones mecánicas
    }
    
    if pd.isna(text) or not str(text).strip():
        return result
    
    # Normalize and tokenize
    text_lower = str(text).lower().strip()
    tokens = re.split(r'[;,/|]+', text_lower)
    tokens = {t.strip() for t in tokens if t.strip()}
    
    # Check each token
    if tokens & FV_TOKENS:
        result['comp_fv'] = True
        result['comp_fv_tv'] = True
    
    if tokens & TV_TOKENS:
        result['comp_tv'] = True
        result['comp_fv_tv'] = True
    
    if tokens & BAV_ALTO_GRADO_TOKENS:
        result['comp_bav_alto_grado'] = True
    
    if tokens & BAV_BAJO_GRADO_TOKENS:
        result['comp_bav_bajo_grado'] = True
    
    if tokens & SHOCK_TOKENS:
        result['comp_shock'] = True
    
    if tokens & IC_TOKENS:
        result['comp_ic'] = True
    
    if tokens & PCR_TOKENS:
        result['comp_pcr'] = True
    
    if tokens & OTRAS_ARRITMIAS_TOKENS:
        result['comp_otras_arritmias'] = True
    
    if tokens & COMP_MECANICAS_TOKENS:
        result['comp_mecanicas'] = True
    
    return result


def process_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Process dataset: parse complicaciones and add only the extracted variables.
    
    Args:
        input_path: Path to input Excel/CSV file
        output_path: Base path for output (without extension)
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"📂 Loading: {input_path}")
    
    # Load
    if input_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
    elif input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path, encoding='utf-8')
    else:
        df = pd.read_parquet(input_path)
    
    logger.info(f"📊 Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for complicaciones column
    if 'complicaciones' not in df.columns:
        raise ValueError("Column 'complicaciones' not found")
    
    # Parse complicaciones
    logger.info("🔄 Parsing 'complicaciones' column...")
    parsed = df['complicaciones'].apply(parse_complicaciones)
    parsed_df = pd.DataFrame(parsed.tolist())
    
    # Add parsed columns to original dataframe
    for col in parsed_df.columns:
        df[col] = parsed_df[col].astype(int)  # Convert bool to int (0/1)
    
    # Statistics
    logger.info("📈 Parsing statistics (RECUIMA-relevant):")
    logger.info(f"   comp_fv_tv (FV o TV): {df['comp_fv_tv'].sum()} ({df['comp_fv_tv'].mean()*100:.1f}%)")
    logger.info(f"   comp_bav_alto_grado: {df['comp_bav_alto_grado'].sum()} ({df['comp_bav_alto_grado'].mean()*100:.1f}%)")
    
    logger.info("📈 Other complications:")
    for col in ['comp_shock', 'comp_ic', 'comp_pcr', 'comp_otras_arritmias', 'comp_mecanicas']:
        logger.info(f"   {col}: {df[col].sum()} ({df[col].mean()*100:.1f}%)")
    
    # Save in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parquet (recommended for ML)
    parquet_path = output_path.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    logger.info(f"💾 Saved: {parquet_path}")
    
    # CSV (for compatibility)
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"💾 Saved: {csv_path}")
    
    # Excel (for manual inspection)
    xlsx_path = output_path.with_suffix('.xlsx')
    df.to_excel(xlsx_path, index=False)
    logger.info(f"💾 Saved: {xlsx_path}")
    
    return df


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "DATA" / "recuima-020425.xlsx"
    output_path = project_root / "DATA" / "recuima-020425-parsed"
    
    # Process
    df = process_dataset(input_path, output_path)
    
    # Summary
    print("\n" + "="*60)
    print("RESUMEN: Variables extraídas de 'complicaciones'")
    print("="*60)
    print(f"\nTotal registros: {len(df)}")
    print(f"Con complicaciones: {df['complicaciones'].notna().sum()}")
    print("\n📌 VARIABLES PARA RECUIMA:")
    print(f"   • comp_fv_tv (FV o TV → 2 pts): {df['comp_fv_tv'].sum()}")
    print(f"   • comp_bav_alto_grado (→ 1 pt): {df['comp_bav_alto_grado'].sum()}")
    print("\n📌 OTRAS COMPLICACIONES:")
    print(f"   • comp_fv: {df['comp_fv'].sum()}")
    print(f"   • comp_tv: {df['comp_tv'].sum()}")
    print(f"   • comp_bav_bajo_grado: {df['comp_bav_bajo_grado'].sum()}")
    print(f"   • comp_shock: {df['comp_shock'].sum()}")
    print(f"   • comp_ic: {df['comp_ic'].sum()}")
    print(f"   • comp_pcr: {df['comp_pcr'].sum()}")
    print(f"   • comp_otras_arritmias: {df['comp_otras_arritmias'].sum()}")
    print(f"   • comp_mecanicas: {df['comp_mecanicas'].sum()}")
    print("\n✅ Dataset guardado en: DATA/recuima-020425-parsed.[parquet|csv|xlsx]")
    
    return df


if __name__ == "__main__":
    main()
