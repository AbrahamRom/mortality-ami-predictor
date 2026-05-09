"""Inverse Optimization and Treatment Recommendation page.

This page allows users to find optimal feature values (e.g., treatment, lifestyle)
that achieve a desired prediction outcome. This is useful for:
- Treatment optimization
- Intervention planning
- "What-if" scenario analysis
- Counterfactual explanations
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app import initialize_state, list_saved_models
from app.config import get_plotly_config
from src.config import CONFIG
from src.explainability import (
    InverseOptimizer,
    plot_optimal_values_comparison,
    plot_confidence_intervals,
    plot_sensitivity_analysis,
    plot_sensitivity_heatmap,
    plot_feature_importance_for_optimization,
    plot_bootstrap_distributions,
    create_optimization_summary_figure,
)

# Initialize
initialize_state()

# Page config
st.set_page_config(
    page_title="Inverse Optimization",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Inverse Optimization & Treatment Recommendations")
st.markdown("""
This tool helps you find **optimal feature values** to achieve a desired prediction.
Perfect for **treatment optimization**, **intervention planning**, and understanding 
**actionable insights** from your models.
""")
st.markdown("---")

# Check if data has been loaded
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    st.success("✅ Using cleaned data")
elif raw_data is not None:
    df = raw_data
    st.warning("⚠️ Using raw data")
else:
    st.warning("⚠️ No data loaded. Please load a dataset in **🧹 Data Cleaning and EDA** first.")
    st.stop()

# Get target column from session state (now stores actual column name)
target_col_name = st.session_state.get('target_column_name', None)

# Determine task for model folder organization
if target_col_name:
    if target_col_name == CONFIG.target_column or target_col_name in ['mortality', 'mortality_inhospital', 'exitus']:
        task = 'mortality'
    elif target_col_name == CONFIG.arrhythmia_column or target_col_name in ['arrhythmia', 'ventricular_arrhythmia']:
        task = 'arrhythmia'
    else:
        task = target_col_name.lower().replace(' ', '_')[:20]
else:
    # Fallback to old behavior
    task = st.session_state.get('target_column', 'mortality')
    if task == 'exitus':
        task = 'mortality'

# ==================== SIDEBAR: Model Selection ====================
st.sidebar.header("🎯 Model Selection")

saved_models = list_saved_models(task)

# If no models found for custom task, also check 'mortality' folder
if not saved_models and task not in ['mortality', 'arrhythmia']:
    st.info(f"ℹ️ No models found for task '{task}', checking 'mortality' folder...")
    saved_models = list_saved_models('mortality')

if not saved_models:
    st.error(f"❌ No trained models found for task '{task}'. Please train models first.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(saved_models.keys()),
    help="Select a trained model for inverse optimization"
)

model_path = saved_models[selected_model_name]

# Load model
try:
    model = joblib.load(model_path)
    st.sidebar.success(f"✅ Model loaded")
except EOFError:
    st.error(f"❌ El archivo del modelo '{selected_model_name}' está corrupto (EOFError). Por favor, elimine el archivo y vuelva a entrenar el modelo.")
    st.info(f"📁 Ruta del archivo: `{model_path}`")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Get feature columns
# Use the actual target column name from session state
target_col = target_col_name if target_col_name else (CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column)

# Exclude all potential target columns from features
excluded_cols = {CONFIG.target_column, CONFIG.arrhythmia_column}
if target_col:
    excluded_cols.add(target_col)
feature_cols = [c for c in df.columns if c not in excluded_cols]

# Check if classifier
is_classifier = hasattr(model, 'predict_proba')

# ==================== SIDEBAR: Optimization Settings ====================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Optimization Settings")

# Target value
if is_classifier:
    target_options = {
        "Survival (Class 0)": 0,
        "Mortality (Class 1)": 1,
    }
    target_label = st.sidebar.selectbox(
        "Desired Outcome",
        list(target_options.keys()),
        help="Select the desired prediction outcome"
    )
    target_value = target_options[target_label]
    
    # For probabilistic interpretation
    use_probability = st.sidebar.checkbox(
        "Optimize Probability",
        value=True,
        help="Optimize predicted probability instead of class label"
    )
    
    if use_probability:
        target_value = st.sidebar.slider(
            "Target Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.1 if target_label.startswith("Survival") else 0.9,
            step=0.05,
            help="Target probability for the positive class (mortality)"
        )
else:
    target_value = st.sidebar.number_input(
        "Target Value",
        value=0.0,
        help="Desired regression target value"
    )

# Optimization method
optimization_method = st.sidebar.selectbox(
    "Optimization Method",
    ["SLSQP", "COBYLA", "differential_evolution"],
    help="""
    - SLSQP: Sequential Least Squares (fast, good for smooth objectives)
    - COBYLA: Constrained Optimization (robust, derivative-free)
    - differential_evolution: Global optimization (slower, finds global optimum)
    """
)

# Number of restarts
n_restarts = st.sidebar.slider(
    "Random Restarts",
    min_value=1,
    max_value=20,
    value=5,
    help="Multiple random initializations to avoid local minima"
)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    tolerance = st.number_input(
        "Convergence Tolerance",
        value=1e-6,
        format="%.1e",
        help="Optimization convergence criterion"
    )
    
    random_seed = st.number_input(
        "Random Seed",
        value=42,
        min_value=0,
        help="For reproducibility"
    )
    
    compute_ci = st.checkbox(
        "Compute Confidence Intervals",
        value=False,
        help="Bootstrap confidence intervals (slower)"
    )
    
    if compute_ci:
        n_bootstrap = st.slider(
            "Bootstrap Iterations",
            min_value=10,
            max_value=100,
            value=30,
            help="Number of bootstrap samples for CI estimation"
        )
        
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Confidence level for intervals"
        )

# ==================== MAIN CONTENT ====================

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Optimization Setup",
    "📊 Results & Analysis",
    "🔍 Sensitivity Analysis",
    "📚 Examples & Help"
])

# ==================== TAB 1: Setup ====================
with tab1:
    st.header("Feature Selection")
    st.markdown("Select which features to optimize and which to keep fixed.")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Modifiable Features")
        st.caption("Features that will be optimized")
        
        # Search functionality
        search_modifiable = st.text_input(
            "Search features",
            key="search_mod",
            placeholder="Type to filter..."
        )
        
        # Filter features
        if search_modifiable:
            filtered_features = [f for f in feature_cols if search_modifiable.lower() in f.lower()]
        else:
            filtered_features = feature_cols
        
        # Select modifiable features
        modifiable_features = st.multiselect(
            "Select features to optimize",
            filtered_features,
            default=[],
            help="These features will be adjusted to achieve the target",
            key="modifiable_features"
        )
        
        if not modifiable_features:
            st.warning("⚠️ Please select at least one modifiable feature")
    
    with col2:
        st.subheader("📌 Fixed Features")
        st.caption("Features that will remain constant")
        
        # Option to set fixed features
        use_fixed = st.checkbox(
            "Set fixed features",
            value=False,
            help="Fix certain features to specific values"
        )
        
        fixed_features = {}
        
        if use_fixed:
            # Select which features to fix
            available_for_fixing = [f for f in feature_cols if f not in modifiable_features]
            
            selected_fixed = st.multiselect(
                "Select features to fix",
                available_for_fixing,
                help="These features will be kept at specified values"
            )
            
            if selected_fixed:
                st.markdown("**Set fixed values:**")
                
                # Create a form for fixed values
                for feat in selected_fixed:
                    if feat in df.columns:
                        feat_median = float(df[feat].median())
                        feat_min = float(df[feat].min())
                        feat_max = float(df[feat].max())
                        
                        fixed_features[feat] = st.number_input(
                            f"{feat}",
                            value=feat_median,
                            min_value=feat_min,
                            max_value=feat_max,
                            help=f"Range: [{feat_min:.2f}, {feat_max:.2f}]",
                            key=f"fixed_{feat}"
                        )
    
    # Initial values section
    st.markdown("---")
    st.subheader("🚀 Initial Values (Optional)")
    
    use_initial = st.checkbox(
        "Provide initial guess",
        value=False,
        help="Starting point for optimization (optional but can help convergence)"
    )
    
    initial_values = {}
    
    if use_initial and modifiable_features:
        st.caption("Set starting values for the optimization")
        
        # Use columns for better layout
        n_cols = 3
        cols = st.columns(n_cols)
        
        for i, feat in enumerate(modifiable_features):
            col_idx = i % n_cols
            
            with cols[col_idx]:
                if feat in df.columns:
                    feat_median = float(df[feat].median())
                    feat_min = float(df[feat].min())
                    feat_max = float(df[feat].max())
                    
                    initial_values[feat] = st.number_input(
                        f"{feat}",
                        value=feat_median,
                        min_value=feat_min,
                        max_value=feat_max,
                        help=f"Median: {feat_median:.2f}",
                        key=f"init_{feat}"
                    )
    
    # Reference patient (optional)
    st.markdown("---")
    st.subheader("👤 Reference Patient (Optional)")
    
    use_reference = st.checkbox(
        "Use reference patient",
        value=False,
        help="Use an existing patient as the starting point"
    )
    
    reference_values = {}
    
    if use_reference:
        patient_idx = st.number_input(
            "Patient Index",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            help="Index of patient in the dataset"
        )
        
        reference_patient = df.iloc[patient_idx]
        
        st.dataframe(
            reference_patient.to_frame().T,
            use_container_width=True
        )
        
        # Use reference for initial values if not set
        if not use_initial:
            for feat in modifiable_features:
                if feat in reference_patient.index:
                    initial_values[feat] = float(reference_patient[feat])
        
        # Use reference for fixed features if not set
        if not use_fixed:
            for feat in feature_cols:
                if feat not in modifiable_features and feat in reference_patient.index:
                    fixed_features[feat] = float(reference_patient[feat])
    
    # Run optimization button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        run_optimization = st.button(
            "🚀 Run Optimization",
            type="primary",
            use_container_width=True,
            disabled=len(modifiable_features) == 0
        )

# ==================== TAB 2: Results ====================
with tab2:
    if run_optimization and modifiable_features:
        # Create optimizer
        with st.spinner("🔄 Initializing optimizer..."):
            optimizer = InverseOptimizer(
                model=model,
                feature_names=feature_cols,
            )
        
        # Run optimization
        with st.spinner(f"🎯 Running optimization with {n_restarts} restarts..."):
            try:
                result = optimizer.optimize(
                    target_value=target_value,
                    modifiable_features=modifiable_features,
                    fixed_features=fixed_features if fixed_features else None,
                    initial_values=initial_values if initial_values else None,
                    reference_data=df,
                    method=optimization_method,
                    n_iterations=n_restarts,
                    random_state=random_seed,
                    tolerance=tolerance,
                    maximize=False,
                )
                
                # Store in session state
                st.session_state.optimization_result = result
                st.session_state.optimization_settings = {
                    'target_value': target_value,
                    'modifiable_features': modifiable_features,
                    'fixed_features': fixed_features,
                    'initial_values': initial_values,
                    'method': optimization_method,
                }
                
                if result['success']:
                    st.success("✅ Optimization completed successfully!")
                else:
                    st.warning(f"⚠️ Optimization finished with warnings: {result.get('optimization_message', 'Unknown')}")
            
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")
                st.exception(e)
                result = None
    
    # Display results if available
    if 'optimization_result' in st.session_state:
        result = st.session_state.optimization_result
        settings = st.session_state.optimization_settings
        
        st.header("📊 Optimization Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Target Value",
                f"{settings['target_value']:.3f}"
            )
        
        with col2:
            st.metric(
                "Achieved Prediction",
                f"{result['achieved_prediction']:.3f}",
                delta=f"{result['achieved_prediction'] - settings['target_value']:.3f}"
            )
        
        with col3:
            st.metric(
                "Success",
                "✅ Yes" if result['success'] else "❌ No"
            )
        
        with col4:
            st.metric(
                "Distance to Target",
                f"{result['distance_to_target']:.6f}"
            )
        
        # Get Plotly configuration
        plotly_config = get_plotly_config()
        
        # Comparison visualization
        st.markdown("---")
        st.subheader("🔄 Original vs Optimal Values")
        
        # Get original values
        original_vals = {}
        if use_reference and 'reference_patient' in locals():
            for feat in modifiable_features:
                if feat in reference_patient.index:
                    original_vals[feat] = float(reference_patient[feat])
        elif initial_values:
            original_vals = initial_values
        else:
            # Use median as "original"
            for feat in modifiable_features:
                if feat in df.columns:
                    original_vals[feat] = float(df[feat].median())
        
        # Create comparison plot
        fig = plot_optimal_values_comparison(
            original_values=original_vals,
            optimal_values=result['optimal_values'],
            feature_names=modifiable_features,
        )
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        # Detailed results table
        st.markdown("---")
        st.subheader("📋 Detailed Optimal Values")
        
        results_df = pd.DataFrame([
            {
                'Feature': feat,
                'Original': original_vals.get(feat, np.nan),
                'Optimal': result['optimal_values'].get(feat, np.nan),
                'Change': result['optimal_values'].get(feat, 0) - original_vals.get(feat, 0),
                'Change %': ((result['optimal_values'].get(feat, 0) - original_vals.get(feat, 0)) / 
                           original_vals.get(feat, 1)) * 100 if original_vals.get(feat, 0) != 0 else 0,
            }
            for feat in modifiable_features
        ])
        
        # Style the dataframe
        st.dataframe(
            results_df.style.format({
                'Original': '{:.4f}',
                'Optimal': '{:.4f}',
                'Change': '{:.4f}',
                'Change %': '{:.2f}%',
            }).background_gradient(subset=['Change'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Comprehensive summary figure
        st.markdown("---")
        st.subheader("📈 Comprehensive Summary")
        
        fig_summary = create_optimization_summary_figure(
            result=result,
            original_values=original_vals
        )
        st.plotly_chart(fig_summary, use_container_width=True, config=plotly_config)
        
        # Confidence intervals (if computed)
        if compute_ci and run_optimization:
            st.markdown("---")
            st.subheader("📊 Confidence Intervals")
            
            with st.spinner("Computing bootstrap confidence intervals..."):
                try:
                    ci_result = optimizer.compute_confidence_intervals(
                        target_value=settings['target_value'],
                        modifiable_features=modifiable_features,
                        fixed_features=fixed_features if fixed_features else None,
                        reference_data=df,
                        n_bootstrap=n_bootstrap,
                        confidence_level=confidence_level,
                        method=optimization_method,
                        random_state=random_seed,
                    )
                    
                    st.session_state.ci_result = ci_result
                    
                    st.success(f"✅ Computed {ci_result['n_successful']}/{ci_result['n_bootstrap']} successful bootstrap iterations")
                
                except Exception as e:
                    st.error(f"❌ Error computing CI: {e}")
                    ci_result = None
        
        # Display CI results
        if 'ci_result' in st.session_state:
            ci_result = st.session_state.ci_result
            
            # CI plot
            fig_ci = plot_confidence_intervals(
                ci_results=ci_result['confidence_intervals'],
                optimal_values=result['optimal_values'],
            )
            st.plotly_chart(fig_ci, use_container_width=True, config=plotly_config)
            
            # CI table
            with st.expander("📊 View Confidence Intervals Table"):
                ci_df = pd.DataFrame([
                    {
                        'Feature': feat,
                        'Mean': data['mean'],
                        'Median': data['median'],
                        'Std': data['std'],
                        'Lower CI': data['lower_ci'],
                        'Upper CI': data['upper_ci'],
                        'CI Width': data['upper_ci'] - data['lower_ci'],
                    }
                    for feat, data in ci_result['confidence_intervals'].items()
                ])
                
                st.dataframe(
                    ci_df.style.format({
                        'Mean': '{:.4f}',
                        'Median': '{:.4f}',
                        'Std': '{:.4f}',
                        'Lower CI': '{:.4f}',
                        'Upper CI': '{:.4f}',
                        'CI Width': '{:.4f}',
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Individual feature distributions
            with st.expander("📈 View Bootstrap Distributions"):
                selected_feat = st.selectbox(
                    "Select feature to visualize",
                    list(ci_result['confidence_intervals'].keys())
                )
                
                fig_dist = plot_bootstrap_distributions(
                    ci_results=ci_result['confidence_intervals'],
                    feature=selected_feat
                )
                st.plotly_chart(fig_dist, use_container_width=True, config=plotly_config)
        
        # Export results
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Export as CSV
            export_df = pd.DataFrame([result['optimal_values']])
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Optimal Values (CSV)",
                data=csv,
                file_name="optimal_values.csv",
                mime="text/csv",
            )
        
        with col_exp2:
            # Export full results as JSON
            import json
            
            export_data = {
                'settings': settings,
                'result': {k: v for k, v in result.items() if k != 'optimal_values'},
                'optimal_values': result['optimal_values'],
            }
            
            # Convert numpy types to native Python types
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            export_data = convert_types(export_data)
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name="optimization_results.json",
                mime="application/json",
            )
    
    else:
        st.info("👈 Configure and run optimization in the **Optimization Setup** tab")

# ==================== TAB 3: Sensitivity Analysis ====================
with tab3:
    if 'optimization_result' in st.session_state:
        result = st.session_state.optimization_result
        settings = st.session_state.optimization_settings
        
        st.header("🔍 Sensitivity Analysis")
        st.markdown("Analyze how changes in optimal values affect the prediction.")
        
        # Settings for sensitivity analysis
        col_sens1, col_sens2 = st.columns(2)
        
        with col_sens1:
            perturbation_percent = st.slider(
                "Perturbation Percentage",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="How much to vary each feature around its optimal value"
            )
        
        with col_sens2:
            n_points = st.slider(
                "Number of Points",
                min_value=10,
                max_value=100,
                value=20,
                step=5,
                help="Resolution of the sensitivity analysis"
            )
        
        if st.button("🔍 Run Sensitivity Analysis", type="primary"):
            with st.spinner("Computing sensitivity..."):
                try:
                    optimizer = InverseOptimizer(
                        model=model,
                        feature_names=feature_cols,
                    )
                    
                    sensitivity_df = optimizer.sensitivity_analysis(
                        optimal_values=result['optimal_values'],
                        modifiable_features=settings['modifiable_features'],
                        fixed_features=settings.get('fixed_features'),
                        perturbation_percent=perturbation_percent,
                        n_points=n_points,
                    )
                    
                    st.session_state.sensitivity_df = sensitivity_df
                    st.success("✅ Sensitivity analysis completed!")
                
                except Exception as e:
                    st.error(f"❌ Error in sensitivity analysis: {e}")
        
        # Display sensitivity results
        if 'sensitivity_df' in st.session_state:
            sensitivity_df = st.session_state.sensitivity_df
            plotly_config = get_plotly_config()
            
            st.markdown("---")
            
            # Line plot
            st.subheader("📈 Sensitivity Curves")
            fig_sens = plot_sensitivity_analysis(
                sensitivity_df=sensitivity_df,
                target_value=settings['target_value'],
            )
            st.plotly_chart(fig_sens, use_container_width=True, config=plotly_config)
            
            # Heatmap
            st.markdown("---")
            st.subheader("🔥 Sensitivity Heatmap")
            fig_heatmap = plot_sensitivity_heatmap(sensitivity_df)
            st.plotly_chart(fig_heatmap, use_container_width=True, config=plotly_config)
            
            # Feature importance for optimization
            st.markdown("---")
            st.subheader("⭐ Feature Importance for Target Achievement")
            st.caption("How much each feature affects the prediction when perturbed")
            
            fig_importance = plot_feature_importance_for_optimization(sensitivity_df)
            st.plotly_chart(fig_importance, use_container_width=True, config=plotly_config)
            
            # Raw data
            with st.expander("📊 View Sensitivity Data"):
                st.dataframe(
                    sensitivity_df.style.format({
                        'value': '{:.4f}',
                        'delta_from_optimal': '{:.4f}',
                        'prediction': '{:.4f}',
                    }),
                    use_container_width=True,
                    hide_index=True
                )
    
    else:
        st.info("👈 Run optimization first to enable sensitivity analysis")

# ==================== TAB 4: Help ====================
with tab4:
    st.header("📚 Guide & Examples")
    
    st.markdown("""
    ## 🎯 What is Inverse Optimization?
    
    **Inverse optimization** finds the optimal feature values to achieve a desired prediction.
    Instead of predicting outcomes from features, it works backwards: given a desired outcome,
    what features should we have?
    
    ### 🏥 Clinical Use Case: Treatment Optimization
    
    **Scenario**: You want to minimize mortality risk for a patient with AMI.
    
    **Steps**:
    1. **Select Model**: Choose your best mortality prediction model
    2. **Set Target**: Select "Survival (Class 0)" or set low mortality probability (e.g., 0.1)
    3. **Choose Modifiable Features**: Select treatments/interventions you can control:
       - Medications (e.g., aspirin, statins, beta-blockers)
       - Clinical interventions (e.g., reperfusion time)
       - Lifestyle factors (if applicable)
    4. **Fix Unchangeable Features**: Lock patient characteristics you cannot change:
       - Age, sex, medical history
    5. **Run Optimization**: Find the optimal treatment combination
    
    ### 📊 Interpreting Results
    
    - **Optimal Values**: The feature values that minimize/maximize the prediction
    - **Change**: How much each feature needs to change from baseline
    - **Achieved Prediction**: The model's prediction using optimal values
    - **Distance to Target**: How close we got to the desired outcome
    
    ### 🔍 Sensitivity Analysis
    
    Shows how "stable" the optimal solution is:
    - **Steep curves**: Small changes have large impact (critical features)
    - **Flat curves**: Robust to variations (flexible features)
    
    ### ⚠️ Important Considerations
    
    1. **Clinical Validity**: Always validate recommendations with domain experts
    2. **Feasibility**: Optimal values might not be practically achievable
    3. **Model Limitations**: Results are only as good as your model
    4. **Confidence Intervals**: Use bootstrap CI to assess uncertainty
    5. **Multiple Solutions**: Different feature combinations may achieve similar outcomes
    
    ### 🧪 Example Scenarios
    """)
    
    with st.expander("📝 Example 1: Reduce Mortality Risk"):
        st.markdown("""
        **Goal**: Find treatment to reduce mortality from 80% to 20%
        
        **Setup**:
        - Target: Probability = 0.2
        - Modifiable: medications, intervention timing
        - Fixed: age, sex, comorbidities
        
        **Expected Output**: Optimal dosages and intervention strategies
        """)
    
    with st.expander("📝 Example 2: Intervention Planning"):
        st.markdown("""
        **Goal**: What's the minimum medication needed to keep mortality < 30%?
        
        **Setup**:
        - Target: Probability = 0.3
        - Modifiable: single medication dose
        - Fixed: all other features at patient values
        
        **Expected Output**: Minimum effective dose
        """)
    
    with st.expander("📝 Example 3: Lifestyle Recommendations"):
        st.markdown("""
        **Goal**: Long-term lifestyle changes to improve prognosis
        
        **Setup**:
        - Target: Survival (Class 0)
        - Modifiable: modifiable risk factors (BP, cholesterol, BMI, etc.)
        - Fixed: demographics, medical history
        
        **Expected Output**: Recommended lifestyle modifications
        """)
    
    st.markdown("---")
    st.markdown("""
    ## 🔬 Scientific Background
    
    This tool implements **constrained optimization** using:
    
    - **SLSQP** (Sequential Least Squares Programming): Efficient for smooth objectives
    - **COBYLA** (Constrained Optimization BY Linear Approximation): Robust derivative-free method
    - **Differential Evolution**: Global optimization for finding best overall solution
    
    **Multiple random restarts** help avoid local optima and find better solutions.
    
    **Bootstrap confidence intervals** quantify uncertainty in optimal values.
    
    ### 📖 References
    
    - Wachter et al. (2017): "Counterfactual Explanations without Opening the Black Box"
    - Ustun et al. (2019): "Actionable Recourse in Linear Classification"
    - SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
    """)
    
    st.markdown("---")
    st.info("""
    💡 **Pro Tip**: Start with a small number of modifiable features and gradually add more.
    This makes optimization faster and results easier to interpret.
    """)

# Footer
st.markdown("---")
st.caption("""
🎯 **Inverse Optimization Module** - Find optimal interventions to achieve desired outcomes.
Always validate results with clinical experts and consider practical feasibility.
""")
