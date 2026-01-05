# 🧬 SFARI Gene Expression Explorer

A sophisticated, Python-based web application for exploring cross-species single-cell RNA-seq gene expression data with a focus on neurodevelopmental genes.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### 🔍 Gene Search & Filtering
- **Multi-species support**: Human, Mouse, Zebrafish, Drosophila
- **Flexible filtering**: By species, dataset, cell type, and timepoint
- **Gene set presets**: Quick access to SFARI risk genes by confidence score
- **Case-insensitive search**: Works with both human and native gene symbols

### 📊 Interactive Heatmaps
- Row-wise Z-score scaling
- Hierarchical clustering (rows and/or columns)
- Multiple color scales
- Faceting by species, dataset, or cell type
- Downloadable expression matrices

### 🔵 Dot Plots
- Size encodes % expressing cells
- Color encodes mean expression
- Flexible grouping options

### 📈 Temporal Dynamics (NEW)
- **Pseudotime trajectories**: Smooth expression curves with confidence intervals
- **Developmental time alignment**: Compare across species using conserved developmental landmarks
- **Pseudotime heatmaps**: Gene expression patterns across developmental progression

### 🔬 Cross-Species Comparison
- Side-by-side expression comparisons
- Ortholog-aware gene matching
- Species-normalized developmental timing

### 📋 Data Export
- Download filtered data as CSV
- Export expression matrices
- Publication-ready figures

## Installation

### Option 1: pip install

```bash
# Clone the repository
git clone https://github.com/your-username/sfari-explorer.git
cd sfari-explorer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda create -n sfari-explorer python=3.10
conda activate sfari-explorer
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

If you have h5ad files, use the data preparation script:

```bash
# Simple usage with a single file
python prepare_data.py \
    --input your_data.h5ad \
    --output ./data/ \
    --species "Human" \
    --cell-type-col "cell_type"

# Using a configuration file (recommended for multiple datasets)
python prepare_data.py --config config.yaml
```

Or if you already have parquet files from the R Shiny version, simply copy them to `./data/`:

```bash
mkdir -p data
cp /path/to/your/parquet/files/*.parquet ./data/
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Data Format

### Required Files (in `./data/` directory)

| File | Description | Required Columns |
|------|-------------|------------------|
| `expression_summaries.parquet` | Aggregated expression data | `species`, `tissue`, `cell_type`, `gene_native`, `gene_human`, `mean_expr`, `pct_expressing`, `n_cells` |
| `celltype_meta.parquet` | Cell type metadata | `species`, `tissue`, `cell_type` |
| `gene_map.parquet` | Gene symbol mappings | `species`, `gene_native`, `gene_human` |
| `risk_genes.parquet` | SFARI risk gene annotations | `gene_symbol`, `gene_score`, `gene_name` |

### Optional Columns for Temporal Analysis

Add these columns to `expression_summaries.parquet` for temporal visualization:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `timepoint` | Original timepoint label | "E12.5", "GW10", "24hpf" |
| `dev_time` | Normalized developmental time (0-1) | 0.15, 0.25, 0.45 |
| `pseudotime` | Pseudotime from trajectory inference | 0.0 - 1.0 |

## Configuration

### Data Preparation Config (`config.yaml`)

```yaml
output_dir: "./data"
ortholog_map_path: "./ortholog_map.csv"  # Optional
risk_genes_path: "./SFARI_genes.csv"

datasets:
  - path: "/path/to/human_brain.h5ad"
    name: "Human Brain Atlas"
    species: "Human"
    cell_type_col: "cell_type"
    timepoint_col: "gestational_week"
    gene_col: "var_names"
    layer: null  # Use .X matrix
    timepoint_mapping:  # Optional: custom time normalization
      "GW8": 0.11
      "GW10": 0.15
      "GW12": 0.19
```

### Ortholog Mapping File

Create a CSV file with cross-species gene mappings:

```csv
species,gene_native,gene_human
Mouse,Shank3,SHANK3
Mouse,Mecp2,MECP2
Zebrafish,shank3a,SHANK3
Drosophila,Shank,SHANK3
```

## Project Structure

```
sfari-explorer/
├── app.py                 # Main Streamlit application
├── temporal.py            # Temporal dynamics module
├── prepare_data.py        # Data preparation pipeline
├── requirements.txt       # Python dependencies
├── config.yaml           # Sample configuration
├── README.md             # This file
└── data/                 # Data directory (create this)
    ├── expression_summaries.parquet
    ├── celltype_meta.parquet
    ├── gene_map.parquet
    └── risk_genes.parquet
```

## Deployment

### Local Development

```bash
streamlit run app.py --server.port 8501
```

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

Note: For large datasets, consider using Streamlit's caching and data compression.

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t sfari-explorer .
docker run -p 8501:8501 -v $(pwd)/data:/app/data sfari-explorer
```

### Heroku / Railway / Render

Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Migrating from R Shiny

If you're migrating from the R Shiny version:

1. **Copy your parquet files** - The data format is compatible:
   ```bash
   cp /path/to/SFARIExplorer/inst/app/data/*.parquet ./data/
   ```

2. **Column name compatibility** - The Python version expects `tissue` column (same as R version).

3. **Add temporal columns** (optional) - Enhance with developmental time data for the new temporal visualization features.

## Comparison: R Shiny vs Python Streamlit

| Feature | R Shiny (Original) | Python Streamlit (New) |
|---------|-------------------|------------------------|
| Heatmap library | ComplexHeatmap | Plotly (interactive) |
| Data loading | arrow::read_parquet | pandas/pyarrow |
| Interactivity | Server-side | Client-side (faster) |
| Deployment | shinyapps.io | Streamlit Cloud, Docker |
| Temporal dynamics | ❌ Not implemented | ✅ Full support |
| Species alignment | ❌ Not implemented | ✅ Developmental time mapping |
| Gene set presets | ❌ Manual entry | ✅ SFARI score presets |
| Dot plots | ❌ Not available | ✅ Interactive |

## Development

### Adding New Features

The modular design makes it easy to extend:

```python
# Add a new visualization in app.py
def create_my_custom_plot(df, genes):
    # Your visualization code
    fig = go.Figure(...)
    return fig

# Add to the tabs section
with tab_new:
    fig = create_my_custom_plot(filtered_df, selected_genes)
    st.plotly_chart(fig)
```

### Running Tests

```bash
pytest tests/
```

## Performance Tips

1. **Data size**: For datasets >1GB, consider:
   - Partitioned parquet files
   - Pre-aggregating by cell type
   - Using DuckDB for queries

2. **Caching**: Streamlit's `@st.cache_data` is used for data loading

3. **Lazy loading**: Consider loading only selected genes/species on demand

## Troubleshooting

### Common Issues

**"No data to display"**
- Check that your parquet files are in the `./data/` directory
- Verify column names match expected format

**Slow performance**
- Reduce number of genes displayed
- Use row/column clustering sparingly for large matrices
- Consider pre-filtering data

**Memory errors**
- Use partitioned parquet files
- Load data lazily by species/dataset

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{sfari_explorer,
  title = {SFARI Gene Expression Explorer},
  author = {Raphael Kubler},
  year = {2026},
  url = {https://github.com/ar-kie/sfari-explorer}
}
```

## Related Resources

- [SFARI Gene Database](https://gene.sfari.org/)
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis in Python
- [scVelo](https://scvelo.readthedocs.io/) - RNA velocity analysis
- [CellxGene](https://cellxgene.cziscience.com/) - Cell atlas browser

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- SFARI (Simons Foundation Autism Research Initiative)
- Single-cell data contributors
- Streamlit and Plotly teams
